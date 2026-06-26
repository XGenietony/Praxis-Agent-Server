// Package process spawns and supervises the backend (llama-server /
// mlx_lm.server) and the frpc tunnel, killing them on shutdown.
package process

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"syscall"
	"time"

	"lmstudio-forward/internal/config"
)

// Manager spawns and supervises the backend and frpc child processes.
// Mirrors src/process.rs.
type Manager struct {
	backendCmd *exec.Cmd
	frpcCmd    *exec.Cmd
}

// NewManager creates an empty Manager.
func NewManager() *Manager {
	return &Manager{}
}

// killPort frees a TCP port by killing whatever process holds it
// (`lsof -ti :PORT` piped into `kill`), then waits one second.
func killPort(port int) {
	out, err := exec.Command("lsof", "-ti", fmt.Sprintf(":%d", port)).Output()
	if err == nil {
		pids := strings.TrimSpace(string(out))
		for _, line := range strings.Split(pids, "\n") {
			pid := strings.TrimSpace(line)
			if pid == "" {
				continue
			}
			_ = exec.Command("kill", pid).Run()
		}
	}
	time.Sleep(1 * time.Second)
}

// waitForHealth polls http://127.0.0.1:port+path once per second up to
// timeoutSecs, returning true as soon as the request succeeds.
func waitForHealth(port int, path string, timeoutSecs int) bool {
	url := fmt.Sprintf("http://127.0.0.1:%d%s", port, path)
	client := &http.Client{Timeout: 2 * time.Second}
	for i := 0; i < timeoutSecs; i++ {
		resp, err := client.Get(url)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode >= 200 && resp.StatusCode < 300 {
				return true
			}
		}
		if i%10 == 9 {
			log.Printf("INFO Still loading... (%ds)", i+1)
		}
		time.Sleep(1 * time.Second)
	}
	return false
}

// Start launches the configured backend then the frpc tunnel.
func (pm *Manager) Start(cfg *config.Config) error {
	if cfg.GgufModel != "" {
		// ── llama-server backend ──
		modelAbs := config.CanonicalizePath(cfg.GgufModel)
		log.Printf("INFO Starting llama-server with model: %s", modelAbs)

		killPort(cfg.BackendPort)

		cmd := exec.Command(cfg.LlamaServer,
			"-m", modelAbs,
			"--port", strconv.Itoa(cfg.BackendPort),
			"-ngl", "99",
			"-c", strconv.Itoa(cfg.CtxSize),
			"-np", "1",
			"-fa", "on",
		)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Start(); err != nil {
			return fmt.Errorf("Failed to start llama-server: %w", err)
		}
		pm.backendCmd = cmd
		log.Printf("INFO llama-server started (PID %d)", cmd.Process.Pid)

		log.Printf("INFO Waiting for llama-server to load model...")
		if waitForHealth(cfg.BackendPort, "/health", 120) {
			log.Printf("INFO llama-server ready")
		} else {
			return fmt.Errorf("llama-server failed to start within 120s")
		}

	} else if cfg.MlxModel != "" {
		// ── mlx_lm.server backend ──
		modelAbs := config.CanonicalizePath(cfg.MlxModel)
		log.Printf("INFO Starting mlx_lm.server with model: %s", modelAbs)

		killPort(cfg.BackendPort)

		// Limit KV cache: ~0.5 KB per token for this model
		cacheBytes := int64(cfg.CtxSize) * 512
		log.Printf("INFO mlx_lm.server prompt-cache-bytes: %d (%d MB)", cacheBytes, cacheBytes/1024/1024)

		args := []string{
			"-m", "mlx_lm", "server",
			"--model", modelAbs,
			"--port", strconv.Itoa(cfg.BackendPort),
			"--prompt-cache-bytes", strconv.FormatInt(cacheBytes, 10),
			"--prompt-cache-size", "1",
			"--max-tokens", strconv.Itoa(cfg.MaxTokens),
			"--temp", strconv.FormatFloat(cfg.Temperature, 'g', -1, 64),
			"--min-p", strconv.FormatFloat(cfg.MinP, 'g', -1, 64),
			"--top-p", strconv.FormatFloat(cfg.TopP, 'g', -1, 64),
			"--prefill-step-size", strconv.Itoa(cfg.PrefillStepSize),
		}
		if cfg.TopK > 0 {
			args = append(args, "--top-k", strconv.Itoa(cfg.TopK))
		}

		cmd := exec.Command(cfg.PythonPath, args...)
		cmd.Env = append(os.Environ(), "PYTHONUNBUFFERED=1")
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Start(); err != nil {
			return fmt.Errorf("Failed to start mlx_lm.server: %w", err)
		}
		pm.backendCmd = cmd
		log.Printf("INFO mlx_lm.server started (PID %d)", cmd.Process.Pid)

		log.Printf("INFO Waiting for mlx_lm.server to load model...")
		if waitForHealth(cfg.BackendPort, "/v1/models", 120) {
			log.Printf("INFO mlx_lm.server ready")
		} else {
			return fmt.Errorf("mlx_lm.server failed to start within 120s")
		}

	} else {
		// ── External backend ──
		log.Printf("INFO Checking backend at 127.0.0.1:%d...", cfg.BackendPort)
		if waitForHealth(cfg.BackendPort, "/v1/models", 5) {
			log.Printf("INFO Backend ready")
		} else {
			log.Printf("WARN Backend not responding — will retry on requests")
		}
	}

	// Start frpc tunnel
	if !cfg.NoFrpc {
		_ = exec.Command("pkill", "-f", "frpc").Run()
		time.Sleep(1 * time.Second)

		log.Printf("INFO Starting frpc tunnel...")
		logFile, err := os.Create("/tmp/frpc.log")
		if err != nil {
			pm.Stop()
			return err
		}

		cmd := exec.Command(cfg.FrpcPath, "-c", cfg.FrpcConfig)
		cmd.Stdout = logFile
		cmd.Stderr = os.Stderr
		if err := cmd.Start(); err != nil {
			pm.Stop()
			return fmt.Errorf("Failed to start frpc: %w", err)
		}
		pm.frpcCmd = cmd
		log.Printf("INFO frpc started (PID %d)", cmd.Process.Pid)

		time.Sleep(2 * time.Second)
		if err := cmd.Process.Signal(syscall.Signal(0)); err != nil {
			pm.Stop()
			return fmt.Errorf("frpc exited during startup: %w", err)
		}
		log.Printf("INFO frpc tunnel ready")
	}

	return nil
}

// Stop kills the spawned child processes. Replaces Rust's Drop impl.
func (pm *Manager) Stop() {
	if pm.backendCmd != nil && pm.backendCmd.Process != nil {
		_ = pm.backendCmd.Process.Kill()
	}
	if pm.frpcCmd != nil && pm.frpcCmd.Process != nil {
		_ = pm.frpcCmd.Process.Kill()
	}
}
