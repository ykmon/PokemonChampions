$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$source = Join-Path $root "launcher\PokemonChampionsAssistantLauncher.cs"
$output = Join-Path $root "PokemonChampionsAssistant.exe"
$csc = Join-Path $env:WINDIR "Microsoft.NET\Framework64\v4.0.30319\csc.exe"

if (-not (Test-Path $csc)) {
    throw "Cannot find C# compiler: $csc"
}

& $csc /nologo /target:winexe /platform:anycpu /out:$output /reference:System.Windows.Forms.dll /reference:System.dll /reference:System.Core.dll $source

Write-Host "Built $output"
