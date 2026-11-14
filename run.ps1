param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArgs
)
& "$PSScriptRoot\venv\Scripts\python.exe" "$PSScriptRoot\main.py" @ScriptArgs
