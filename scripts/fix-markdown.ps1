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

    # 1. Jekyll layout: 필드 제거
    if ($content -match '(?m)^layout:') {
        $content = [regex]::Replace($content, '(?m)^layout:.*(\r?\n)', '')
        $changes += 'layout: 필드 제거'
    }

    # 2. **bold**한글조사 패턴에 U+200B 삽입
    # .NET regex 에서 한글 범위: \uAC00-\uD7A3
    $zws = [char]0x200B
    $re = New-Object System.Text.RegularExpressions.Regex('([^\s\*])(\*\*)([\uAC00-\uD7A3])')
    $hits = $re.Matches($content)
    if ($hits.Count -gt 0) {
        $content = $re.Replace($content, {
            param($mx)
            $mx.Groups[1].Value + $zws + $mx.Groups[2].Value + $mx.Groups[3].Value
        })
        $changes += "Bold+조사 $($hits.Count)곳 수정 (ZWS 삽입)"
    }

    if ($changes.Count -eq 0) {
        Write-Host "  OK  이미 정상 - 수정 없음" -ForegroundColor Green
        return
    }
    foreach ($c in $changes) { Write-Host "  >> $c" -ForegroundColor Yellow }
    if ($DryRun) { Write-Host "  -- DryRun - 저장 안함" -ForegroundColor Cyan; return }
    [System.IO.File]::WriteAllText($FilePath, $content, [System.Text.Encoding]::UTF8)
    Write-Host "  OK  저장 완료" -ForegroundColor Green
}

Write-Host ""
Write-Host "====================================" -ForegroundColor DarkCyan
Write-Host " Hugo Blog Markdown Fixer" -ForegroundColor Cyan
if ($DryRun) { Write-Host " [DryRun 미리보기]" -ForegroundColor Yellow }
Write-Host "====================================" -ForegroundColor DarkCyan

if ($File) {
    Repair-MarkdownFile -FilePath $File
} elseif ($Dir) {
    $files = Get-ChildItem -Path $Dir -Filter "*.md" -Recurse
    Write-Host "  $($files.Count)개 파일 처리 중..." -ForegroundColor Cyan
    foreach ($f in $files) { Repair-MarkdownFile -FilePath $f.FullName }
} else {
    Write-Host ""
    Write-Host "사용법:" -ForegroundColor Yellow
    Write-Host "  .\scripts\fix-markdown.ps1 -File <파일경로>"
    Write-Host "  .\scripts\fix-markdown.ps1 -Dir <폴더경로>"
    Write-Host "  옵션: -DryRun (미리보기, 저장 안함)"
    Write-Host ""
}
Write-Host ""
