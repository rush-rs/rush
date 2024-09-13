with import <nixpkgs> {
  crossSystem = {
    config = "s390x-unknown-linux-gnu";
  };
};

mkShell {
  # buildInputs = [ zlib ]; # your dependencies here
shellHook = ''
    # if running from zsh, reenter zsh
    if [[ $(ps -e | grep $PPID) == *"zsh" ]]; then
    export SHELL=zsh
    zsh
    exit
    fi
'';
}
