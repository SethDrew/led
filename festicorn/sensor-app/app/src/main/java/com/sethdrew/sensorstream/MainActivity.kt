package com.sethdrew.sensorstream

import android.Manifest
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.View
import android.view.WindowManager
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.SeekBar
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.sethdrew.sensorstream.databinding.ActivityMainBinding
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetAddress
import kotlin.concurrent.thread

data class Installation(val name: String, val ip: String, val mac: String) {
    override fun toString(): String = "$name ($ip)"
}

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var prefs: SharedPreferences
    private val handler = Handler(Looper.getMainLooper())

    private var lastSliderSendMs = 0L
    private val SLIDER_THROTTLE_MS = 500L

    private val installations = mutableListOf<Installation>()
    private lateinit var spinnerAdapter: ArrayAdapter<Installation>
    private var selectedInstallation: Installation? = null

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
        binding.intervalInput.setText(prefs.getInt("interval_ms", 40).toString())

        spinnerAdapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, installations)
        spinnerAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        binding.installationSpinner.adapter = spinnerAdapter
        binding.installationSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, pos: Int, id: Long) {
                selectedInstallation = installations[pos]
                prefs.edit().putString("last_ip", selectedInstallation?.ip).apply()
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {
                selectedInstallation = null
            }
        }

        val savedIp = prefs.getString("last_ip", null)
        if (savedIp != null) {
            installations.add(Installation("last session", savedIp, ""))
            spinnerAdapter.notifyDataSetChanged()
            selectedInstallation = installations[0]
        }

        binding.scanBtn.setOnClickListener { scanForInstallations() }

        // Effect buttons
        binding.fxGravity.setOnClickListener { sendEffect('g') }
        binding.fxSparkle.setOnClickListener { sendEffect('s') }
        binding.fxFireMeld.setOnClickListener { sendEffect('m') }
        binding.fxFlicker.setOnClickListener { sendEffect('f') }
        binding.fxBloom.setOnClickListener { sendEffect('b') }
        binding.fxRainbow.setOnClickListener { sendEffect('i') }
        binding.fxNebula.setOnClickListener { sendEffect('n') }
        binding.fxOff.setOnClickListener { sendEffect('x') }

        // Toggles
        binding.micToggle.setOnCheckedChangeListener { _, isChecked ->
            updateSensitivityState()
            if (isChecked) {
                requestMicPermissionAndEnable()
            } else {
                sendToggleUpdate()
            }
        }
        binding.gyroToggle.setOnCheckedChangeListener { _, _ -> sendToggleUpdate() }

        // Sliders
        binding.brightnessSlider.setOnSeekBarChangeListener(throttledSlider('B'))
        binding.sensitivitySlider.setOnSeekBarChangeListener(throttledSlider('S'))
        binding.speedSlider.setOnSeekBarChangeListener(throttledSlider('V'))

        if (Build.VERSION.SDK_INT >= 33) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this,
                    arrayOf(Manifest.permission.POST_NOTIFICATIONS), 100)
            }
        }
    }

    private fun sendEffect(cmd: Char) {
        sendCommand(cmd)
        updateSliderStates(cmd)
        if (cmd != 'x') {
            startService()
        }
    }

    private fun updateSliderStates(cmd: Char) {
        val speedEnabled = cmd == 'n' || cmd == 'i'
        binding.speedSlider.isEnabled = speedEnabled
        binding.speedLabel.alpha = if (speedEnabled) 1.0f else 0.4f
        updateSensitivityState()
    }

    private fun updateSensitivityState() {
        val enabled = binding.micToggle.isChecked
        binding.sensitivitySlider.isEnabled = enabled
        binding.sensitivityLabel.alpha = if (enabled) 1.0f else 0.4f
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

    private fun getSelectedHost(): String? = selectedInstallation?.ip

    private fun startService() {
        val host = getSelectedHost() ?: return
        val interval = binding.intervalInput.text.toString().toLongOrNull() ?: 40L
        prefs.edit()
            .putString("last_ip", host)
            .putInt("interval_ms", interval.toInt())
            .apply()

        val intent = Intent(this, SensorService::class.java).apply {
            action = SensorService.ACTION_START
            putExtra(SensorService.EXTRA_HOST, host)
            putExtra(SensorService.EXTRA_PORT, 4210)
            putExtra(SensorService.EXTRA_INTERVAL_MS, interval)
            putExtra(SensorService.EXTRA_MIC_ENABLED, binding.micToggle.isChecked)
            putExtra(SensorService.EXTRA_GYRO_ENABLED, binding.gyroToggle.isChecked)
        }
        ContextCompat.startForegroundService(this, intent)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }

    private fun scanForInstallations() {
        binding.scanBtn.isEnabled = false
        binding.scanBtn.text = "..."
        thread {
            val found = mutableListOf<Installation>()
            try {
                val sock = DatagramSocket()
                sock.broadcast = true
                sock.soTimeout = 3000
                val msg = "ROAD?".toByteArray()
                val broadcast = InetAddress.getByName("255.255.255.255")
                sock.send(DatagramPacket(msg, msg.size, broadcast, 4212))

                val deadline = System.currentTimeMillis() + 3000
                while (System.currentTimeMillis() < deadline) {
                    try {
                        val buf = ByteArray(128)
                        val resp = DatagramPacket(buf, buf.size)
                        sock.soTimeout = (deadline - System.currentTimeMillis()).toInt().coerceAtLeast(100)
                        sock.receive(resp)
                        val reply = String(resp.data, 0, resp.length)
                        if (reply.startsWith("ROAD!")) {
                            val payload = reply.substring(5)
                            val ip = resp.address.hostAddress ?: continue
                            val parts = payload.split("|", limit = 2)
                            val name: String
                            val mac: String
                            if (parts.size == 2) {
                                name = parts[0]
                                mac = parts[1]
                            } else {
                                name = ip
                                mac = parts[0]
                            }
                            if (found.none { it.ip == ip }) {
                                found.add(Installation(name, ip, mac))
                            }
                        }
                    } catch (_: Exception) {
                        break
                    }
                }
                sock.close()
            } catch (_: Exception) {}

            runOnUiThread {
                installations.clear()
                if (found.isNotEmpty()) {
                    installations.addAll(found)
                    binding.scanBtn.text = "${found.size} found"
                } else {
                    binding.scanBtn.text = "None"
                }
                spinnerAdapter.notifyDataSetChanged()
                if (installations.isNotEmpty()) {
                    binding.installationSpinner.setSelection(0)
                    selectedInstallation = installations[0]
                }
                handler.postDelayed({
                    binding.scanBtn.isEnabled = true
                    binding.scanBtn.text = "Scan"
                }, 2000)
            }
        }
    }

    private fun throttledSlider(cmd: Char) = object : SeekBar.OnSeekBarChangeListener {
        override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
            val now = System.currentTimeMillis()
            if (now - lastSliderSendMs >= SLIDER_THROTTLE_MS) {
                lastSliderSendMs = now
                sendSlider(cmd, progress)
            }
        }
        override fun onStartTrackingTouch(seekBar: SeekBar?) {}
        override fun onStopTrackingTouch(seekBar: SeekBar?) {
            sendSlider(cmd, seekBar?.progress ?: 128)
        }
    }

    private fun sendSlider(cmd: Char, value: Int) {
        val host = getSelectedHost() ?: return
        thread {
            try {
                val sock = DatagramSocket()
                sock.soTimeout = 1000
                val addr = InetAddress.getByName(host)
                val data = byteArrayOf(cmd.code.toByte(), value.toByte())
                sock.send(DatagramPacket(data, 2, addr, 4211))
                sock.close()
            } catch (_: Exception) {}
        }
    }

    private fun sendCommand(cmd: Char) {
        val host = getSelectedHost() ?: return
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
