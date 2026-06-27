package config

import (
	"strings"
	"testing"
)

func validConfig() Config {
	return Config{
		BackendPort:                    1234,
		ServerPort:                     5678,
		CtxSize:                        8192,
		MaxTokens:                      1024,
		MaxRequestBodyBytes:            1024,
		BackendHealthTimeoutSeconds:    2,
		ServerReadHeaderTimeoutSeconds: 5,
		ServerIdleTimeoutSeconds:       120,
		ShutdownTimeoutSeconds:         10,
		RagStepTimeoutSeconds:          120,
		RagFailureMode:                 "closed",
	}
}

func TestValidateRejectsMutuallyExclusiveBackends(t *testing.T) {
	cfg := validConfig()
	cfg.MlxModel = "mlx"
	cfg.GgufModel = "model.gguf"

	err := cfg.Validate()
	if err == nil || !strings.Contains(err.Error(), "mutually exclusive") {
		t.Fatalf("want mutually exclusive backend error, got %v", err)
	}
}

func TestValidateRejectsInvalidRAGSettings(t *testing.T) {
	cfg := validConfig()
	cfg.RagEnabled = true
	cfg.QdrantURL = "://bad"
	cfg.EmbedURL = "http://embed"
	cfg.QdrantCollection = ""
	cfg.EmbedModel = "embed"
	cfg.EmbedDim = 2
	cfg.RagTopK = 1
	cfg.RagMaxRounds = 1
	cfg.RagChunkSize = 100
	cfg.RagChunkOverlap = 100

	err := cfg.Validate()
	if err == nil {
		t.Fatal("want validation error")
	}
	msg := err.Error()
	for _, want := range []string{"qdrant-url", "qdrant-collection", "rag-chunk-overlap"} {
		if !strings.Contains(msg, want) {
			t.Fatalf("validation error %q missing %q", msg, want)
		}
	}
}

func TestValidateAcceptsFailOpenMode(t *testing.T) {
	cfg := validConfig()
	cfg.RagFailureMode = "open"
	if err := cfg.Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}
}
