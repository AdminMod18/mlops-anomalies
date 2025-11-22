#!/bin/bash
for i in {1..10000}
do
curl -X POST http://127.0.0.1:34615/predict \
  -H "Content-Type: application/json" \
  -d '{"values":[1.2, -0.4, 3.1, 0.5, 2.0, -1.1, 4.5, 3.3, 0.8, -0.6]}'
echo ""
sleep 1
done
