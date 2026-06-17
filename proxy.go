package main

import (
	"net"
	"net/http"
	"strings"
)

// AppState holds the shared dependencies passed to every HTTP handler.
// Mirrors Rust's `proxy::AppState`.
type AppState struct {
	Config     Config
	HTTPClient *http.Client
	Rag        *RagClient // nil when RAG disabled (RagClient defined in rag.go)
}

// getClientIP extracts the originating client IP from a request.
// It prefers the first segment of the X-Forwarded-For header, falling back to
// the host portion of r.RemoteAddr, then "unknown".
func getClientIP(r *http.Request) string {
	if forwarded := r.Header.Get("X-Forwarded-For"); forwarded != "" {
		// split on ',', take the first segment, trim whitespace.
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

// checkAPIKey reports whether the request is authorized.
// An empty expectedKey disables auth (always authorized). Otherwise the request
// must carry either "Authorization: Bearer <key>" or "x-api-key: <key>".
func checkAPIKey(r *http.Request, expectedKey string) bool {
	if expectedKey == "" {
		return true
	}
	if auth := r.Header.Get("Authorization"); auth != "" {
		if auth == "Bearer "+expectedKey {
			return true
		}
	}
	if key := r.Header.Get("x-api-key"); key != "" {
		if key == expectedKey {
			return true
		}
	}
	return false
}
