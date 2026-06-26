package proxy

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestCheckAPIKeyAcceptsNormalizedBearerAndXAPIKey(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	req.Header.Set("Authorization", "bearer   secret")
	if !CheckAPIKey(req, " secret ") {
		t.Fatal("expected normalized bearer token to pass")
	}

	req = httptest.NewRequest(http.MethodGet, "/", nil)
	req.Header.Set("x-api-key", "secret")
	if !CheckAPIKey(req, "secret") {
		t.Fatal("expected x-api-key to pass")
	}
}

func TestCheckAPIKeyRejectsMalformedBearer(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	req.Header.Set("Authorization", "Bearer secret extra")
	if CheckAPIKey(req, "secret") {
		t.Fatal("expected malformed bearer header to fail")
	}
}

func TestReadLimitedBodyRejectsOversizedBodies(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/", strings.NewReader("abcdef"))
	rec := httptest.NewRecorder()
	_, err := ReadLimitedBody(rec, req, 3)
	if err == nil {
		t.Fatal("expected oversized body error")
	}
}
