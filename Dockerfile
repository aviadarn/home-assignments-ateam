FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY code/requirements-test.txt ./requirements-test.txt
RUN pip install --no-cache-dir -r requirements-test.txt

# Copy source code
COPY code/ ./code/

# Default: run both tests in sequence
# Test 1 — end-to-end learning loop script (prints CTR before/after)
# Test 2 — pytest suite (19 unit/integration tests)
CMD ["sh", "-c", \
  "echo '=== Test 1: Learning Loop End-to-End ===' && \
   python3 code/src/learning_loop.py && \
   echo '' && \
   echo '=== Test 2: Pytest Suite ===' && \
   python3 -m pytest code/tests/ -v"]
