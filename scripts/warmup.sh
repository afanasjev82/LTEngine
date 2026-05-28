#!/usr/bin/env bash
set -e
curl -s -X POST http://localhost:5050/translate \
  -H "Content-Type: application/json" \
  -d '{"q":"hello world","source":"en","target":"es","format":"text"}'
echo
