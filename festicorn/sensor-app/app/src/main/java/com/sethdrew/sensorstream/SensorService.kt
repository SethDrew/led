package com.sethdrew.sensorstream

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import androidx.core.app.NotificationCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetAddress
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.sqrt

class SensorService : Service(), SensorEventListener {

    companion object {
        const val TAG = "Zergling"
        const val CHANNEL_ID = "sensor_stream_channel"
        const val NOTIF_ID = 1
        const val EXTRA_HOST = "host"
        const val EXTRA_PORT = "port"
        const val EXTRA_INTERVAL_MS = "interval_ms"
        const val EXTRA_MIC_ENABLED = "mic_enabled"
        const val EXTRA_GYRO_ENABLED = "gyro_enabled"
        const val ACTION_START = "start"
        const val ACTION_STOP = "stop"
        const val ACTION_TOGGLE = "toggle"

        const val PACKET_SIZE = 15
        const val ONSET_PORT = 4213

        val running = AtomicLong(0L)
        val pktsSent = AtomicInteger(0)
        val samplesTotal = AtomicInteger(0)
        val lastStatus = AtomicReference<String>("idle")
    }

    private lateinit var sensorManager: SensorManager
    private var accel: Sensor? = null
    private var gyro: Sensor? = null

    private var wakeLock: PowerManager.WakeLock? = null
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private var senderJob: Job? = null

    private var socket: DatagramSocket? = null
    private var targetAddr: InetAddress? = null
    private var targetPort: Int = 4210
    private var intervalMs: Long = 40L

    @Volatile private var ax: Float = 0f
    @Volatile private var ay: Float = 0f
    @Volatile private var az: Float = 0f
    @Volatile private var gx: Float = 0f
    @Volatile private var gy: Float = 0f
    @Volatile private var gz: Float = 0f

    @Volatile private var micEnabled: Boolean = false
    @Volatile private var gyroEnabled: Boolean = true
    @Volatile private var currentRms: Int = 0

    private var audioRecord: AudioRecord? = null
    private var micJob: Job? = null

    private val sendBuf = ByteArray(PACKET_SIZE)
    private val bb: ByteBuffer = ByteBuffer.wrap(sendBuf).order(ByteOrder.LITTLE_ENDIAN)

    private val onsetBuf = ByteArray(4)
    private var onsetFloor: Float = 0f
    private var prevRms: Float = 0f
    private var onsetCooldownMs: Long = 0L

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        accel = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyro = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        createChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_STOP -> {
                stopStreaming()
                stopForeground(STOP_FOREGROUND_REMOVE)
                stopSelf()
                return START_NOT_STICKY
            }
            ACTION_TOGGLE -> {
                micEnabled = intent.getBooleanExtra(EXTRA_MIC_ENABLED, false)
                gyroEnabled = intent.getBooleanExtra(EXTRA_GYRO_ENABLED, true)
                updateMicCapture()
                return START_STICKY
            }
            else -> {
                val host = intent?.getStringExtra(EXTRA_HOST) ?: "192.168.86.127"
                targetPort = intent?.getIntExtra(EXTRA_PORT, 4210) ?: 4210
                intervalMs = intent?.getLongExtra(EXTRA_INTERVAL_MS, 40L) ?: 40L
                micEnabled = intent?.getBooleanExtra(EXTRA_MIC_ENABLED, false) ?: false
                gyroEnabled = intent?.getBooleanExtra(EXTRA_GYRO_ENABLED, true) ?: true
                startForeground(NOTIF_ID, buildNotification("→ $host:$targetPort"))
                startStreaming(host)
            }
        }
        return START_STICKY
    }

    private fun createChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val nm = getSystemService(NotificationManager::class.java)
            val ch = NotificationChannel(CHANNEL_ID, "Zergling Stream", NotificationManager.IMPORTANCE_LOW)
            ch.description = "Background IMU streaming"
            nm.createNotificationChannel(ch)
        }
    }

    private fun buildNotification(text: String): Notification {
        val pi = PendingIntent.getActivity(
            this, 0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Zergling UDP")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_menu_compass)
            .setOngoing(true)
            .setContentIntent(pi)
            .build()
    }

    private fun updateNotification(text: String) {
        val nm = getSystemService(NotificationManager::class.java)
        nm.notify(NOTIF_ID, buildNotification(text))
    }

    private fun updateMicCapture() {
        if (micEnabled && audioRecord == null) {
            startMicCapture()
        } else if (!micEnabled) {
            stopMicCapture()
            currentRms = 0
        }
    }

    private fun startMicCapture() {
        val sampleRate = 16000
        val bufSize = AudioRecord.getMinBufferSize(
            sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
        ).coerceAtLeast(1024)

        try {
            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                bufSize
            )
            audioRecord?.startRecording()
        } catch (e: SecurityException) {
            Log.e(TAG, "mic permission denied", e)
            micEnabled = false
            return
        }

        micJob = scope.launch {
            val buf = ShortArray(bufSize / 2)
            while (micEnabled) {
                val read = audioRecord?.read(buf, 0, buf.size) ?: -1
                if (read > 0) {
                    var sumSq = 0.0
                    for (i in 0 until read) {
                        val s = buf[i].toDouble()
                        sumSq += s * s
                    }
                    val rms = sqrt(sumSq / read).toFloat()
                    currentRms = rms.toInt().coerceIn(0, 65535)

                    if (rms < onsetFloor) {
                        onsetFloor += 0.01f * (rms - onsetFloor)
                    } else {
                        onsetFloor += 0.001f * (rms - onsetFloor)
                    }
                    if (onsetFloor < 50f) onsetFloor = 50f

                    val now = System.currentTimeMillis()
                    if (rms > onsetFloor * 2.0f
                        && rms - prevRms > onsetFloor * 0.5f
                        && now >= onsetCooldownMs) {
                        val strength = ((rms - onsetFloor) / (onsetFloor * 10f)).coerceIn(0f, 1f)
                        val strengthU8 = (strength * 255f).toInt().coerceIn(0, 255)
                        onsetBuf[0] = 0xAA.toByte()
                        onsetBuf[1] = strengthU8.toByte()
                        onsetBuf[2] = 0x00.toByte()
                        onsetBuf[3] = 0x55.toByte()
                        val addr = targetAddr
                        val sock = socket
                        if (addr != null && sock != null) {
                            try {
                                sock.send(DatagramPacket(onsetBuf, 4, addr, ONSET_PORT))
                            } catch (_: Exception) {}
                        }
                        onsetCooldownMs = now + 120L
                    }
                    prevRms = rms
                }
            }
        }
    }

    private fun stopMicCapture() {
        micJob?.cancel()
        micJob = null
        try {
            audioRecord?.stop()
            audioRecord?.release()
        } catch (_: Exception) {}
        audioRecord = null
    }

    private fun startStreaming(host: String) {
        if (running.get() != 0L) {
            scope.launch {
                targetAddr = InetAddress.getByName(host)
            }
            updateMicCapture()
            return
        }
        running.set(System.currentTimeMillis())
        pktsSent.set(0); samplesTotal.set(0)
        lastStatus.set("starting")

        val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "Zergling:wl").apply {
            setReferenceCounted(false)
            acquire(12 * 60 * 60 * 1000L)
        }

        accel?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST) }
        gyro?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST) }

        updateMicCapture()

        senderJob = scope.launch {
            try {
                targetAddr = InetAddress.getByName(host)
                socket = DatagramSocket()
                lastStatus.set("sending → $host:$targetPort")
            } catch (e: Exception) {
                Log.e(TAG, "socket setup failed", e)
                lastStatus.set("setup err: ${e.message}")
                return@launch
            }

            var lastUiUpdateMs = 0L
            while (running.get() != 0L) {
                packAndSend()
                pktsSent.incrementAndGet()
                val now = System.currentTimeMillis()
                if (now - lastUiUpdateMs > 1000) {
                    lastUiUpdateMs = now
                    val elapsed = (now - running.get()) / 1000.0
                    val rate = if (elapsed > 0) pktsSent.get() / elapsed else 0.0
                    updateNotification("sent=${pktsSent.get()} (${"%.1f".format(rate)}/s)")
                }
                delay(intervalMs)
            }
            try { socket?.close() } catch (_: Exception) {}
            socket = null
        }
        Log.i(TAG, "started udp → $host:$targetPort interval=${intervalMs}ms")
    }

    private fun packAndSend() {
        val addr = targetAddr ?: return
        val sock = socket ?: return
        val accelScale = 16384.0f / 9.81f
        val gyroScale = 131.0f * 180.0f / Math.PI.toFloat()
        val axR: Short
        val ayR: Short
        val azR: Short
        val gxR: Short
        val gyR: Short
        val gzR: Short
        if (gyroEnabled) {
            axR = clampShort(ax * accelScale)
            ayR = clampShort(ay * accelScale)
            azR = clampShort(az * accelScale)
            gxR = clampShort(gx * gyroScale)
            gyR = clampShort(gy * gyroScale)
            gzR = clampShort(gz * gyroScale)
        } else {
            axR = 0; ayR = 0; azR = 0
            gxR = 0; gyR = 0; gzR = 0
        }
        bb.clear()
        bb.putShort(axR); bb.putShort(ayR); bb.putShort(azR)
        bb.putShort(gxR); bb.putShort(gyR); bb.putShort(gzR)
        if (micEnabled) {
            bb.putShort(currentRms.toShort())
            bb.put(1)
        } else {
            bb.putShort(0)
            bb.put(0)
        }
        sock.send(DatagramPacket(sendBuf, PACKET_SIZE, addr, targetPort))
    }

    private fun clampShort(v: Float): Short {
        val i = v.toInt()
        return when {
            i > Short.MAX_VALUE.toInt() -> Short.MAX_VALUE
            i < Short.MIN_VALUE.toInt() -> Short.MIN_VALUE
            else -> i.toShort()
        }
    }

    private fun stopStreaming() {
        running.set(0L)
        senderJob?.cancel()
        senderJob = null
        sensorManager.unregisterListener(this)
        stopMicCapture()
        try { socket?.close() } catch (_: Exception) {}
        socket = null
        wakeLock?.let { if (it.isHeld) it.release() }
        wakeLock = null
        lastStatus.set("stopped")
        Log.i(TAG, "stopped")
    }

    override fun onDestroy() {
        super.onDestroy()
        stopStreaming()
        scope.cancel()
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                ax = event.values[0]; ay = event.values[1]; az = event.values[2]
            }
            Sensor.TYPE_GYROSCOPE -> {
                gx = event.values[0]; gy = event.values[1]; gz = event.values[2]
            }
            else -> return
        }
        samplesTotal.incrementAndGet()
    }
}
