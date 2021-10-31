{ devshell, texlive }: 

devshell.mkShell {
packages = [texlive.combined.scheme-medium];
commands = [ { package = "latexrun"; } ];
}
