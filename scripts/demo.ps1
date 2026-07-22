# Demo script (Windows PowerShell): query → feedback → bandit state
$Base = if ($env:BASE_URL) { $env:BASE_URL } else { "http://127.0.0.1:8000" }
$Headers = @{ "Content-Type" = "application/json" }
if ($env:API_KEY) { $Headers["X-API-Key"] = $env:API_KEY }
if ($env:ADMIN_API_KEY) { $AdminHeaders = @{ "X-API-Key" = $env:ADMIN_API_KEY } }
elseif ($env:API_KEY) { $AdminHeaders = @{ "X-API-Key" = $env:API_KEY } }
else { $AdminHeaders = @{} }

Write-Host "== health =="
Invoke-RestMethod "$Base/health" | ConvertTo-Json

Write-Host "== query =="
$queryBody = @{ query = "How do I fix DNS resolution failures on VPN?" } | ConvertTo-Json
$resp = Invoke-RestMethod -Method POST -Uri "$Base/query" -Headers $Headers -Body $queryBody
$resp | ConvertTo-Json -Depth 6
$txn = $resp.transaction_id

Write-Host "== feedback =="
$fb = @{ transaction_id = $txn; feedback_score = 1 } | ConvertTo-Json
Invoke-RestMethod -Method POST -Uri "$Base/feedback" -Headers $Headers -Body $fb | ConvertTo-Json -Depth 6

Write-Host "== tool query =="
$toolBody = @{ query = "Is payments-down.internal up right now?" } | ConvertTo-Json
Invoke-RestMethod -Method POST -Uri "$Base/query" -Headers $Headers -Body $toolBody | ConvertTo-Json -Depth 6

Write-Host "== rl state =="
Invoke-RestMethod "$Base/rl/state" -Headers $AdminHeaders | ConvertTo-Json -Depth 8
