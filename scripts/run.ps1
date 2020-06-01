param (
  [Parameter(Mandatory=$true)][string]$taskName
)
python "$taskName.py"