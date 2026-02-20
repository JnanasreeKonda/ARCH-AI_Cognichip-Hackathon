# PowerShell script to set API keys from .env file
# Usage: .\set_api_keys.ps1

Write-Host "Setting API keys from .env file..." -ForegroundColor Green

if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
            $key = $matches[1].Trim()
            $value = $matches[2].Trim()
            [Environment]::SetEnvironmentVariable($key, $value, "Process")
            Write-Host "  ✓ Set $key" -ForegroundColor Cyan
        }
    }
    Write-Host "`nAPI keys loaded! You can now run: python main.py" -ForegroundColor Green
} else {
    Write-Host "⚠️  .env file not found!" -ForegroundColor Yellow
    Write-Host "Creating .env file template..." -ForegroundColor Yellow
    @"
# OpenAI API Key
OPENAI_API_KEY=your-key-here

# Optional: Other API keys
# ANTHROPIC_API_KEY=your-anthropic-key-here
# GEMINI_API_KEY=your-gemini-key-here
"@ | Out-File -FilePath ".env" -Encoding UTF8
    Write-Host "Created .env file. Please add your API keys." -ForegroundColor Yellow
}
