package com.sethdrew.sensorstream

import android.Manifest
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.WindowManager
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.sethdrew.sensorstream.databinding.ActivityMainBinding
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetAddress
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var prefs: SharedPreferences
    private val handler = Handler(Looper.getMainLooper())

    private val ticker = object : Runnable {
        override fun run() {
            updateStatus()
            handler.postDelayed(this, 500)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        prefs = getSharedPreferences("sensorstream", MODE_PRIVATE)
        binding.hostInput.setText(prefs.getString("host", "192.168.86.127"))
        binding.portInput.setText(prefs.getInt("port", 4210).toString())
        binding.intervalInput.setText(prefs.getInt("interval_ms", 40).toString())

        binding.startBtn.setOnClickListener { startService() }
        binding.stopBtn.setOnClickListener { stopService() }
        binding.discoverBtn.setOnClickListener { discoverEsp() }

        binding.fxGravity.setOnClickListener { sendCommand('g') }
        binding.fxSparkle.setOnClickListener { sendCommand('s') }
        binding.fxFireMeld.setOnClickListener { sendCommand('m') }
        binding.fxFlicker.setOnClickListener { sendCommand('f') }
        binding.fxBloom.setOnClickListener { sendCommand('b') }
        binding.fxSparkleSimple.setOnClickListener { sendCommand('y') }

        binding.micToggle.setOnCheckedChangeListener { _, isChecked ->
            if (isChecked) {
                requestMicPermissionAndEnable()
            } else {
                sendToggleUpdate()
            }
        }
        binding.gyroToggle.setOnCheckedChangeListener { _, _ -> sendToggleUpdate() }

        if (Build.VERSION.SDK_INT >= 33) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this,
                    arrayOf(Manifest.permission.POST_NOTIFICATIONS), 100)
            }
        }
    }

    private fun requestMicPermissionAndEnable() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                arrayOf(Manifest.permission.RECORD_AUDIO), 101)
        } else {
            sendToggleUpdate()
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 101) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                sendToggleUpdate()
            } else {
                binding.micToggle.isChecked = false
            }
        }
    }

    private fun sendToggleUpdate() {
        val intent = Intent(this, SensorService::class.java).apply {
            action = SensorService.ACTION_TOGGLE
            putExtra(SensorService.EXTRA_MIC_ENABLED, binding.micToggle.isChecked)
            putExtra(SensorService.EXTRA_GYRO_ENABLED, binding.gyroToggle.isChecked)
        }
        startService(intent)
    }

    override fun onResume() {
        super.onResume()
        handler.post(ticker)
    }

    override fun onPause() {
        super.onPause()
        handler.removeCallbacks(ticker)
    }

    private fun startService() {
        val host = binding.hostInput.text.toString().trim()
        val port = binding.portInput.text.toString().toIntOrNull() ?: 4210
        val interval = binding.intervalInput.text.toString().toLongOrNull() ?: 40L
        prefs.edit()
            .putString("host", host)
            .putInt("port", port)
            .putInt("interval_ms", interval.toInt())
            .apply()

        val intent = Intent(this, SensorService::class.java).apply {
            action = SensorService.ACTION_START
            putExtra(SensorService.EXTRA_HOST, host)
            putExtra(SensorService.EXTRA_PORT, port)
            putExtra(SensorService.EXTRA_INTERVAL_MS, interval)
            putExtra(SensorService.EXTRA_MIC_ENABLED, binding.micToggle.isChecked)
            putExtra(SensorService.EXTRA_GYRO_ENABLED, binding.gyroToggle.isChecked)
        }
        ContextCompat.startForegroundService(this, intent)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }

    private fun stopService() {
        val intent = Intent(this, SensorService::class.java).apply {
            action = SensorService.ACTION_STOP
        }
        startService(intent)
        window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }

    private fun discoverEsp() {
        binding.discoverBtn.isEnabled = false
        binding.discoverBtn.text = "..."
        thread {
            try {
                val sock = DatagramSocket()
                sock.broadcast = true
                sock.soTimeout = 2000
                val msg = "ROAD?".toByteArray()
                val broadcast = InetAddress.getByName("255.255.255.255")
                sock.send(DatagramPacket(msg, msg.size, broadcast, 4212))
                val buf = ByteArray(64)
                val resp = DatagramPacket(buf, buf.size)
                sock.receive(resp)
                val reply = String(resp.data, 0, resp.length)
                if (reply.startsWith("ROAD!")) {
                    val ip = resp.address.hostAddress ?: ""
                    val mac = reply.substring(5)
                    runOnUiThread {
                        binding.hostInput.setText(ip)
                        binding.discoverBtn.text = "Found"
                    }
                }
                sock.close()
            } catch (_: Exception) {
                runOnUiThread { binding.discoverBtn.text = "N/A" }
            }
            runOnUiThread {
                handler.postDelayed({
                    binding.discoverBtn.isEnabled = true
                    binding.discoverBtn.text = "Find"
                }, 2000)
            }
        }
    }

    private fun sendCommand(cmd: Char) {
        val host = binding.hostInput.text.toString().trim()
        thread {
            try {
                val sock = DatagramSocket()
                sock.soTimeout = 1000
                val addr = InetAddress.getByName(host)
                val data = byteArrayOf(cmd.code.toByte())
                sock.send(DatagramPacket(data, 1, addr, 4211))
                val buf = ByteArray(64)
                val resp = DatagramPacket(buf, buf.size)
                sock.receive(resp)
                val reply = String(resp.data, 0, resp.length)
                if (reply.startsWith("FX:")) {
                    val fx = reply.substring(3)
                    runOnUiThread {
                        binding.statusText.text = "effect: $fx"
                    }
                }
                sock.close()
            } catch (_: Exception) {}
        }
    }

    private fun updateStatus() {
        val startedAt = SensorService.running.get()
        val running = startedAt != 0L
        val sent = SensorService.pktsSent.get()
        val samples = SensorService.samplesTotal.get()
        val last = SensorService.lastStatus.get()
        val elapsed = if (running) (System.currentTimeMillis() - startedAt) / 1000.0 else 0.0
        val rate = if (elapsed > 0) sent / elapsed else 0.0
        binding.statusText.text = buildString {
            append("running: ").append(running).append('\n')
            append("pkts: ").append(sent).append(" (")
                .append(String.format("%.1f", rate)).append("/s)\n")
            append("sensors: ").append(samples).append('\n')
            append(last)
        }
    }
}
