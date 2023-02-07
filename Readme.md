
Getting data into notebook is tricky:

Suggestion below doesn't work
https://help.valohai.com/hc/en-us/articles/4422921110929-Add-input-files-to-your-execution

# Get the location of Valohai inputs directory
VH_INPUTS_DIR = os.getenv('VH_INPUTS_DIR', '.inputs')
 
# Get the path to your individual inputs file
# e.g. /valohai/inputs/<input-name>/<filename.ext<
path_to_file = os.path.join(VH_INPUTS_DIR, 'myinput/mydata.csv')
 
pd.read_csv(path_to_file)


Hardcoded paths. 