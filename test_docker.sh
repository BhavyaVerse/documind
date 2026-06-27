BASE_URL="${BASE_URL:-http://localhost:8000}"
PASS=0
FAIL=0
 
#  Helpers 
 
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'   # No colour
 
pass() { echo -e "${GREEN}   PASS${NC}  $1"; ((PASS++)); }
fail() { echo -e "${RED}   FAIL${NC}  $1"; ((FAIL++)); }
info() { echo -e "${YELLOW}  →${NC}  $1"; }
 
#  Test functions 
 
test_health() {
    info "GET /health"
    RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}/health")
    if [ "$RESPONSE" = "200" ]; then
        pass "/health returned 200"
    else
        fail "/health returned ${RESPONSE} (expected 200)"
    fi
 
    # Parse the JSON body
    BODY=$(curl -s "${BASE_URL}/health")
    info "Response: ${BODY}"
}
 
test_docs_endpoint() {
    info "GET /docs (Swagger UI)"
    RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}/docs")
    if [ "$RESPONSE" = "200" ]; then
        pass "/docs returned 200"
    else
        fail "/docs returned ${RESPONSE}"
    fi
}
 
test_query_amazon() {
    info "POST /query — Amazon revenue question"
    BODY=$(curl -s -X POST "${BASE_URL}/query" \
        -H "Content-Type: application/json" \
        -d '{
            "question": "What are Amazons primary business segments?",
            "top_k": 3,
            "run_validation": false
        }')
 
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "${BASE_URL}/query" \
        -H "Content-Type: application/json" \
        -d '{"question": "What are Amazons primary business segments?", "top_k": 3, "run_validation": false}')
 
    if [ "$HTTP_CODE" = "200" ]; then
        pass "POST /query returned 200"
        ANSWER=$(echo "$BODY" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('answer','')[:120])" 2>/dev/null)
        CITATIONS=$(echo "$BODY" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('citations_used',[]))" 2>/dev/null)
        LATENCY=$(echo "$BODY" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('latency_ms',0))" 2>/dev/null)
        info "Answer preview : ${ANSWER}..."
        info "Citations used : ${CITATIONS}"
        info "Latency        : ${LATENCY} ms"
    else
        fail "POST /query returned ${HTTP_CODE}"
        info "Response body: ${BODY}"
    fi
}
 
test_query_apple() {
    info "POST /query — Apple services question"
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "${BASE_URL}/query" \
        -H "Content-Type: application/json" \
        -d '{"question": "How does Apples Services segment generate revenue?", "top_k": 3, "run_validation": false}')
 
    if [ "$HTTP_CODE" = "200" ]; then
        pass "POST /query (Apple) returned 200"
    else
        fail "POST /query (Apple) returned ${HTTP_CODE}"
    fi
}
 
test_query_visa() {
    info "POST /query — Visa business model question"
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "${BASE_URL}/query" \
        -H "Content-Type: application/json" \
        -d '{"question": "How does Visa earn revenue from its payment network?", "top_k": 3, "run_validation": false}')
 
    if [ "$HTTP_CODE" = "200" ]; then
        pass "POST /query (Visa) returned 200"
    else
        fail "POST /query (Visa) returned ${HTTP_CODE}"
    fi
}
 
test_short_question_validation() {
    info "POST /query — input too short (expect 422)"
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "${BASE_URL}/query" \
        -H "Content-Type: application/json" \
        -d '{"question": "hi"}')
 
    if [ "$HTTP_CODE" = "422" ]; then
        pass "Short input correctly rejected with 422"
    else
        fail "Expected 422 for short input, got ${HTTP_CODE}"
    fi
}
 
test_with_validation() {
    info "POST /query — with faithfulness validation enabled"
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "${BASE_URL}/query" \
        -H "Content-Type: application/json" \
        -d '{
            "question": "Who is the CEO of Morgan Stanley?",
            "top_k": 3,
            "run_validation": true
        }')
 
    if [ "$HTTP_CODE" = "200" ]; then
        pass "POST /query with validation returned 200"
    else
        fail "POST /query with validation returned ${HTTP_CODE}"
    fi
}
 
#  Run all tests 
 
echo ""
echo "============================================================"
echo "  DocuMind — Docker Smoke Tests"
echo "  Target: ${BASE_URL}"
echo "============================================================"
 
echo ""
echo "[ 1 / 6 ]  Health check"
test_health
 
echo ""
echo "[ 2 / 6 ]  Swagger UI"
test_docs_endpoint
 
echo ""
echo "[ 3 / 6 ]  Query: Amazon segments (no validation)"
test_query_amazon
 
echo ""
echo "[ 4 / 6 ]  Query: Apple services (no validation)"
test_query_apple
 
echo ""
echo "[ 5 / 6 ]  Query: Visa revenue (no validation)"
test_query_visa
 
echo ""
echo "[ 6 / 6 ]  Query: Morgan Stanley CEO (with validation)"
test_with_validation
 
echo ""
echo "[ Bonus ]   Input validation (expect 422)"
test_short_question_validation
 
# ── Summary ───────────────────────────────────────────────────────────────────
 
echo ""
echo "============================================================"
echo -e "  Results:  ${GREEN}${PASS} passed${NC}  |  ${RED}${FAIL} failed${NC}"
echo "============================================================"
echo ""
 
if [ $FAIL -gt 0 ]; then
    echo "Some tests failed. Check the container logs:"
    echo "  docker compose logs -f documind"
    exit 1
else
    echo "All tests passed. Your DocuMind container is working correctly."
    echo ""
    echo "  Swagger UI  →  ${BASE_URL}/docs"
    echo "  Health      →  ${BASE_URL}/health"
    exit 0
fi
 