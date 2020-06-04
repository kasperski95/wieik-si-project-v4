param (
  [Parameter(Mandatory=$true)][string]$taskName
)
python "$taskName.py" --word2vec-limit 1000000