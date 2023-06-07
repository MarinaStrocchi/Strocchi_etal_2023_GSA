# To clean up and re-install everything from scratch
# This is needed is you want to pull the latest
# version of gpytGPE or Historia for instance

pip freeze | grep -v '@' | xargs pip uninstall -y
pip uninstall gpytGPE
pip uninstall Historia
pip uninstall GSA_library
