// Package proxy holds the shared application state and request-level helpers
// (client IP extraction, API-key auth) used across all HTTP handlers.
package proxy

import (
	"crypto/subtle"
	"io"
	"net"
	"net/http"
	"strings"

	"lmstudio-forward/internal/config"
	"lmstudio-forward/internal/rag"
)

// AppState holds the shared dependencies passed to every HTTP handler.
// Mirrors Rust's `proxy::AppState`.
type AppState struct {
	Config     config.Config
	HTTPClient *http.Client
	Rag        *rag.Client // nil when RAG disabled
}

// GetClientIP extracts the originating client IP from a request. It prefers the
// first segment of the X-Forwarded-For header, falling back to the host portion
// of r.RemoteAddr, then "unknown".
func GetClientIP(r *http.Request) string {
	if forwarded := r.Header.Get("X-Forwarded-For"); forwarded != "" {
		first := forwarded
		if idx := strings.IndexByte(forwarded, ','); idx >= 0 {
			first = forwarded[:idx]
		}
		return strings.TrimSpace(first)
	}
	if r.RemoteAddr != "" {
		if host, _, err := net.SplitHostPort(r.RemoteAddr); err == nil {
			return host
		}
		return r.RemoteAddr
	}
	return "unknown"
}

// CheckAPIKey reports whether the request is authorized. An empty expectedKey
// disables auth (always authorized). Otherwise the request must carry either
// "Authorization: Bearer <key>" or "x-api-key: <key>".
func CheckAPIKey(r *http.Request, expectedKey string) bool {
	expected := strings.TrimSpace(expectedKey)
	if expected == "" {
		return true
	}
	if token := bearerToken(r.Header.Get("Authorization")); token != "" && secureCompare(token, expected) {
		return true
	}
	if token := strings.TrimSpace(r.Header.Get("x-api-key")); token != "" && secureCompare(token, expected) {
		return true
	}
	return false
}

func bearerToken(header string) string {
	parts := strings.Fields(header)
	if len(parts) != 2 || !strings.EqualFold(parts[0], "Bearer") {
		return ""
	}
	return strings.TrimSpace(parts[1])
}

func secureCompare(a, b string) bool {
	if len(a) != len(b) {
		return false
	}
	return subtle.ConstantTimeCompare([]byte(a), []byte(b)) == 1
}

// ReadLimitedBody caps request bodies before reading them fully into memory.
func ReadLimitedBody(w http.ResponseWriter, r *http.Request, maxBytes int64) ([]byte, error) {
	if maxBytes <= 0 {
		return io.ReadAll(r.Body)
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxBytes)
	return io.ReadAll(r.Body)
}
