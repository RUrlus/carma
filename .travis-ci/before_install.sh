#!/usr/bin/env bash
set -eo pipefail
[[ "$DEBUG_CI" == true ]] && set -x

case $TRAVIS_OS_NAME in
  linux|osx)    
    ;;
  windows)
    choco install --no-progress -y python --version 3.7
    choco install --no-progress -y curl
    ;;
  *)
    echo "Unknown OS [$TRAVIS_OS_NAME]"
    exit 1
    ;;
esac
