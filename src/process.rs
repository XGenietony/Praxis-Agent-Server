use std::process::Stdio;
use tokio::process::{Child, Command};
use tracing::info;
use anyhow::{Result, Context};
use crate::config::Config;

pub struct ProcessManager {
    frpc_pid: Option<u32>,
    _frpc_child: Option<Child>,
    backend_pid: Option<u32>,
    _backend_child: Option<Child>,
}

impl ProcessManager {
    pub fn new() -> Self {
        Self {
            frpc_pid: None,
            _frpc_child: None,
            backend_pid: None,
            _backend_child: None,
        }
    }

    async fn kill_port(&self, port: u16) {
        let _ = Command::new("lsof")
            .args(["-ti", &format!(":{}", port)])
            .output()
            .await
            .map(|o| {
                let pids = String::from_utf8_lossy(&o.stdout);
                for pid in pids.trim().lines() {
                    let _ = std::process::Command::new("kill").arg(pid.trim()).output();
                }
            });
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
    }

    async fn wait_for_health(&self, port: u16, path: &str, timeout_secs: u64) -> bool {
        for i in 0..timeout_secs {
            if reqwest::get(format!("http://127.0.0.1:{}{}", port, path)).await.is_ok() {
                return true;
            }
            if i % 10 == 9 {
                info!("Still loading... ({}s)", i + 1);
            }
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }
        false
    }

    pub async fn start(&mut self, config: &Config) -> Result<()> {
        if let Some(ref gguf_path) = config.gguf_model {
            // ── llama-server backend ──
            let model_abs = gguf_path.canonicalize()
                .unwrap_or_else(|_| gguf_path.clone());
            info!("Starting llama-server with model: {}", model_abs.display());

            self.kill_port(config.backend_port).await;

            let child = Command::new(&config.llama_server)
                .args([
                    "-m", &model_abs.to_string_lossy(),
                    "--port", &config.backend_port.to_string(),
                    "-ngl", "99",
                    "-c", &config.ctx_size.to_string(),
                    "-np", "1",
                    "-fa", "on",
                ])
                .stdout(Stdio::inherit())
                .stderr(Stdio::inherit())
                .kill_on_drop(true)
                .spawn()
                .context("Failed to start llama-server")?;

            let pid = child.id().unwrap();
            self.backend_pid = Some(pid);
            self._backend_child = Some(child);
            info!("llama-server started (PID {})", pid);

            info!("Waiting for llama-server to load model...");
            if self.wait_for_health(config.backend_port, "/health", 120).await {
                info!("llama-server ready");
            } else {
                anyhow::bail!("llama-server failed to start within 120s");
            }

        } else if let Some(ref model_path) = config.mlx_model {
            // ── mlx_lm.server backend ──
            let model_abs = model_path.canonicalize()
                .unwrap_or_else(|_| model_path.clone());
            info!("Starting mlx_lm.server with model: {}", model_abs.display());

            self.kill_port(config.backend_port).await;

            // Limit KV cache: ~0.5 KB per token for this model
            let cache_bytes = (config.ctx_size as u64) * 512;
            let cache_bytes_str = cache_bytes.to_string();
            let max_tokens_str = config.max_tokens.to_string();
            let temp_str = config.temperature.to_string();
            let min_p_str = config.min_p.to_string();
            let top_p_str = config.top_p.to_string();
            let prefill_str = config.prefill_step_size.to_string();
            let port_str = config.backend_port.to_string();
            let model_str = model_abs.to_string_lossy().to_string();
            let top_k_str = config.top_k.to_string();
            info!("mlx_lm.server prompt-cache-bytes: {} ({} MB)", cache_bytes, cache_bytes / 1024 / 1024);

            let mut args = vec![
                "-m", "mlx_lm", "server",
                "--model", &model_str,
                "--port", &port_str,
                "--prompt-cache-bytes", &cache_bytes_str,
                "--prompt-cache-size", "1",
                "--max-tokens", &max_tokens_str,
                "--temp", &temp_str,
                "--min-p", &min_p_str,
                "--top-p", &top_p_str,
                "--prefill-step-size", &prefill_str,
            ];
            if config.top_k > 0 {
                args.extend_from_slice(&["--top-k", &top_k_str]);
            }

            let child = Command::new(&config.python_path)
                .args(&args)
                .env("PYTHONUNBUFFERED", "1")
                .stdout(Stdio::inherit())
                .stderr(Stdio::inherit())
                .kill_on_drop(true)
                .spawn()
                .context("Failed to start mlx_lm.server")?;

            let pid = child.id().unwrap();
            self.backend_pid = Some(pid);
            self._backend_child = Some(child);
            info!("mlx_lm.server started (PID {})", pid);

            info!("Waiting for mlx_lm.server to load model...");
            if self.wait_for_health(config.backend_port, "/v1/models", 120).await {
                info!("mlx_lm.server ready");
            } else {
                anyhow::bail!("mlx_lm.server failed to start within 120s");
            }

        } else {
            // ── External backend ──
            info!("Checking backend at 127.0.0.1:{}...", config.backend_port);
            if self.wait_for_health(config.backend_port, "/v1/models", 5).await {
                info!("Backend ready");
            } else {
                tracing::warn!("Backend not responding — will retry on requests");
            }
        }

        // Start frpc tunnel
        if !config.no_frpc {
            let _ = Command::new("pkill").arg("-f").arg("frpc").output().await;
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

            info!("Starting frpc tunnel...");
            let frpc_child = Command::new(&config.frpc_path)
                .arg("-c").arg(&config.frpc_config)
                .stdout(Stdio::from(std::fs::File::create("/tmp/frpc.log")?))
                .stderr(Stdio::inherit())
                .kill_on_drop(true)
                .spawn()
                .context("Failed to start frpc")?;

            self.frpc_pid = Some(frpc_child.id().unwrap());
            self._frpc_child = Some(frpc_child);
            info!("frpc started (PID {})", self.frpc_pid.unwrap());
            tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
            info!("frpc tunnel ready");
        }

        Ok(())
    }
}

impl Drop for ProcessManager {
    fn drop(&mut self) {
        if let Some(pid) = self.backend_pid {
            let _ = std::process::Command::new("kill").arg(pid.to_string()).output();
        }
        if let Some(pid) = self.frpc_pid {
            let _ = std::process::Command::new("kill").arg(pid.to_string()).output();
        }
    }
}
