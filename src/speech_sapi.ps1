param([Parameter(Mandatory=$true)][string]$Jobs)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
    $batch = Get-Content -LiteralPath $Jobs -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($job in $batch) {
        if ($job.voice) { $synth.SelectVoice([string]$job.voice) }
        $synth.Rate = [int]$job.rate
        $synth.SetOutputToWaveFile([string]$job.output, $format)
        # Speak plain text, never interpret customer content as SSML or shell code.
        $synth.Speak([string]$job.text)
        $synth.SetOutputToNull()
    }
}
finally { $synth.Dispose() }
