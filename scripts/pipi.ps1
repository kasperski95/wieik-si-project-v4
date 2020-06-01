param (
  [Parameter()][AllowNull()][string]$package
)

if ([String]::IsNullOrWhiteSpace($package)) {
  pip install -r requirements.txt
} Else {
  pip install $package && 
  pip freeze > requirements.txt
}
