# Test script for API endpoints

# Test health endpoint
Write-Host "Testing health endpoint..."
$healthResponse = Invoke-RestMethod -Uri "http://localhost:8001/health" -Method Get
Write-Host "Health Response: $($healthResponse | ConvertTo-Json)"

# Test video analysis endpoint
Write-Host "`nTesting video analysis endpoint..."
$videoPath = "sample_chicken_video.mp4"
$uri = "http://localhost:8001/analyze_video"

# Create multipart form data
$boundary = [System.Guid]::NewGuid().ToString()
$LF = "`r`n"

$bodyLines = (
    "--$boundary",
    "Content-Disposition: form-data; name=`"video`"; filename=`"sample_chicken_video.mp4`"",
    "Content-Type: video/mp4$LF",
    [System.IO.File]::ReadAllBytes($videoPath),
    "--$boundary",
    "Content-Disposition: form-data; name=`"fps_sample`"$LF",
    "2",
    "--$boundary",
    "Content-Disposition: form-data; name=`"conf_thresh`"$LF", 
    "0.5",
    "--$boundary--$LF"
) -join $LF

try {
    $response = Invoke-RestMethod -Uri $uri -Method Post -Body $bodyLines -ContentType "multipart/form-data; boundary=$boundary"
    Write-Host "Analysis Response: $($response | ConvertTo-Json -Depth 10)"
} catch {
    Write-Host "Error: $($_.Exception.Message)"
    Write-Host "Response: $($_.Exception.Response)"
}