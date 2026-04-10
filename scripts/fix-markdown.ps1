param(
    [string]$File,
    [string]$Dir,
    [switch]$DryRun
)
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

function Repair-MarkdownFile {
    param([string]$FilePath)
    if (-not (Test-Path $FilePath)) {
        Write-Host "  X: $FilePath" -ForegroundColor Red; return
    }
    $fname = Split-Path $FilePath -Leaf
    Write-Host ""
    Write-Host "[ $fname ]" -ForegroundColor White
    $content = [System.IO.File]::ReadAllText($FilePath, [System.Text.Encoding]::UTF8)
    $changes = @()

    # 1. Remove Jekyll layout: field
    if ($content -match '(?m)^layout:') {
        $content = [regex]::Replace($content, '(?m)^layout:.*(\r?\n)', '')
        $changes += 'Removed layout: field'
    }

    # 2. Insert U+200B before closing ** when followed by Korean syllable
    # .NET regex Korean range: \uAC00-\uD7A3
    $zws = [char]0x200B
    $re = New-Object System.Text.RegularExpressions.Regex('([^\s\*])(\*\*)([\uAC00-\uD7A3])')
    $hits = $re.Matches($content)
    if ($hits.Count -gt 0) {
        $content = $re.Replace($content, {
            param($mx)
            $mx.Groups[1].Value + $zws + $mx.Groups[2].Value + $mx.Groups[3].Value
        })
        $changes += "Bold+particle: $($hits.Count) location(s) fixed (ZWS inserted)"
    }

    if ($changes.Count -eq 0) {
        Write-Host "  OK  No changes needed" -ForegroundColor Green
        return
    }
    foreach ($c in $changes) { Write-Host "  >> $c" -ForegroundColor Yellow }
    if ($DryRun) { Write-Host "  -- DryRun: not saved" -ForegroundColor Cyan; return }
    [System.IO.File]::WriteAllText($FilePath, $content, [System.Text.Encoding]::UTF8)
    Write-Host "  OK  Saved" -ForegroundColor Green
}

Write-Host ""
Write-Host "====================================" -ForegroundColor DarkCyan
Write-Host " Hugo Blog Markdown Fixer" -ForegroundColor Cyan
if ($DryRun) { Write-Host " [DryRun mode]" -ForegroundColor Yellow }
Write-Host "====================================" -ForegroundColor DarkCyan

if ($File) {
    Repair-MarkdownFile -FilePath $File
} elseif ($Dir) {
    $files = Get-ChildItem -Path $Dir -Filter "*.md" -Recurse
    Write-Host "  Processing $($files.Count) file(s)..." -ForegroundColor Cyan
    foreach ($f in $files) { Repair-MarkdownFile -FilePath $f.FullName }
} else {
    Write-Host ""
    Write-Host "Usage:" -ForegroundColor Yellow
    Write-Host "  .\scripts\fix-markdown.ps1 -File <filepath>"
    Write-Host "  .\scripts\fix-markdown.ps1 -Dir <dirpath>"
    Write-Host "  Option: -DryRun (preview only, no save)"
    Write-Host ""
}
Write-Host ""
