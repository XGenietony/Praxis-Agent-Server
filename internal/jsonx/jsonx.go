// Package jsonx provides the dynamic-JSON navigation layer that mirrors Rust's
// `serde_json::Value`. JSON is decoded into `any`, where:
//
//	object -> map[string]any
//	array  -> []any
//	string -> string
//	number -> float64
//	bool   -> bool
//	null   -> nil
//
// All helpers are nil-safe: navigating through a missing/wrong-typed value
// yields the zero value instead of panicking. This is the single shared
// foundation every other package builds on.
package jsonx

import (
	"encoding/json"
	"log"
	"net/http"
	"strconv"
)

// Parse unmarshals bytes into a dynamic value.
func Parse(b []byte) (any, error) {
	var v any
	err := json.Unmarshal(b, &v)
	return v, err
}

// Marshal marshals v, returning nil on error.
func Marshal(v any) []byte {
	b, err := json.Marshal(v)
	if err != nil {
		return nil
	}
	return b
}

// MarshalString marshals v to a string, returning "" on error.
func MarshalString(v any) string {
	return string(Marshal(v))
}

// MarshalPretty marshals v with 2-space indentation (mirrors serde_json's
// to_string_pretty), returning nil on error.
func MarshalPretty(v any) []byte {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return nil
	}
	return b
}

// MarshalStrict marshals v and returns the underlying encoding error. Use this
// at protocol and HTTP boundaries where silent empty output would corrupt the
// response.
func MarshalStrict(v any) ([]byte, error) {
	return json.Marshal(v)
}

// WriteJSON writes a JSON response, falling back to a minimal 500 response when
// marshaling fails before any bytes are sent.
func WriteJSON(w http.ResponseWriter, status int, v any) {
	b, err := MarshalStrict(v)
	if err != nil {
		log.Printf("ERROR JSON marshal failed: %v", err)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error":"failed to marshal JSON response"}`))
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write(b)
}

// AsObj returns v as a JSON object, or nil if v is not an object.
func AsObj(v any) map[string]any {
	m, _ := v.(map[string]any)
	return m
}

// AsArr returns v as a JSON array, or nil if v is not an array.
func AsArr(v any) []any {
	a, _ := v.([]any)
	return a
}

// AsStr returns (string, ok).
func AsStr(v any) (string, bool) {
	s, ok := v.(string)
	return s, ok
}

// Str returns v as a string, or "" if v is not a string.
func Str(v any) string {
	s, _ := v.(string)
	return s
}

// Bool returns v as a bool, or false.
func Bool(v any) bool {
	b, _ := v.(bool)
	return b
}

// Num returns (float64, ok).
func Num(v any) (float64, bool) {
	f, ok := v.(float64)
	return f, ok
}

// Get returns object field `key`, or nil if v is not an object / key absent.
func Get(v any, key string) any {
	m, ok := v.(map[string]any)
	if !ok {
		return nil
	}
	return m[key]
}

// GetStr returns object field `key` as a string, or "".
func GetStr(v any, key string) string { return Str(Get(v, key)) }

// GetArr returns object field `key` as an array, or nil.
func GetArr(v any, key string) []any { return AsArr(Get(v, key)) }

// GetObj returns object field `key` as an object, or nil.
func GetObj(v any, key string) map[string]any { return AsObj(Get(v, key)) }

// GetBool returns object field `key` as a bool, or false.
func GetBool(v any, key string) bool { return Bool(Get(v, key)) }

// GetNum returns object field `key` as (float64, ok).
func GetNum(v any, key string) (float64, bool) { return Num(Get(v, key)) }

// Has reports whether v is an object containing key.
func Has(v any, key string) bool {
	m, ok := v.(map[string]any)
	if !ok {
		return false
	}
	_, present := m[key]
	return present
}

// Pointer walks nested object keys and array indices. A numeric path segment
// indexes into an array; any other segment indexes into an object. Returns nil
// if any step misses. Mirrors serde_json's `.pointer("/a/0/b")`.
func Pointer(v any, path ...string) any {
	cur := v
	for _, p := range path {
		switch c := cur.(type) {
		case map[string]any:
			cur = c[p]
		case []any:
			idx, err := strconv.Atoi(p)
			if err != nil || idx < 0 || idx >= len(c) {
				return nil
			}
			cur = c[idx]
		default:
			return nil
		}
	}
	return cur
}
