package main

import (
	"encoding/json"
	"strconv"
)

// jsonx provides the dynamic-JSON navigation layer that mirrors Rust's
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
// contract every other file builds on — do not change these signatures.

// parseJSON unmarshals bytes into a dynamic value.
func parseJSON(b []byte) (any, error) {
	var v any
	err := json.Unmarshal(b, &v)
	return v, err
}

// toJSON marshals v, returning nil on error.
func toJSON(v any) []byte {
	b, err := json.Marshal(v)
	if err != nil {
		return nil
	}
	return b
}

// toJSONString marshals v to a string, returning "" on error.
func toJSONString(v any) string {
	return string(toJSON(v))
}

// toJSONPretty marshals v with 2-space indentation (mirrors serde_json's
// to_string_pretty), returning nil on error.
func toJSONPretty(v any) []byte {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return nil
	}
	return b
}

// asObj returns v as a JSON object, or nil if v is not an object.
func asObj(v any) map[string]any {
	m, _ := v.(map[string]any)
	return m
}

// asArr returns v as a JSON array, or nil if v is not an array.
func asArr(v any) []any {
	a, _ := v.([]any)
	return a
}

// asStr returns (string, ok).
func asStr(v any) (string, bool) {
	s, ok := v.(string)
	return s, ok
}

// str returns v as a string, or "" if v is not a string.
func str(v any) string {
	s, _ := v.(string)
	return s
}

// boolv returns v as a bool, or false.
func boolv(v any) bool {
	b, _ := v.(bool)
	return b
}

// numv returns (float64, ok).
func numv(v any) (float64, bool) {
	f, ok := v.(float64)
	return f, ok
}

// get returns object field `key`, or nil if v is not an object / key absent.
func get(v any, key string) any {
	m, ok := v.(map[string]any)
	if !ok {
		return nil
	}
	return m[key]
}

// getStr returns object field `key` as a string, or "".
func getStr(v any, key string) string { return str(get(v, key)) }

// getArr returns object field `key` as an array, or nil.
func getArr(v any, key string) []any { return asArr(get(v, key)) }

// getObj returns object field `key` as an object, or nil.
func getObj(v any, key string) map[string]any { return asObj(get(v, key)) }

// getBool returns object field `key` as a bool, or false.
func getBool(v any, key string) bool { return boolv(get(v, key)) }

// getNum returns object field `key` as (float64, ok).
func getNum(v any, key string) (float64, bool) { return numv(get(v, key)) }

// has reports whether v is an object containing key.
func has(v any, key string) bool {
	m, ok := v.(map[string]any)
	if !ok {
		return false
	}
	_, present := m[key]
	return present
}

// pointer walks nested object keys and array indices. A numeric path segment
// indexes into an array; any other segment indexes into an object. Returns nil
// if any step misses. Mirrors serde_json's `.pointer("/a/0/b")`.
func pointer(v any, path ...string) any {
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
