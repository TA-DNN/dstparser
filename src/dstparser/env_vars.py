import subprocess

def script_vars(shell_script, script_args=""):
    """
    Compare environment variables before and after running a shell script
    and extract added and changed variables.

    Args:
        shell_script (str): The path to the shell script you want to execute.
        script_args (str): The arguments to pass to the shell script.

    Returns:
        dict: A dictionary containing added and changed environment variables with their values.

    Example:
        added_vars = script_vars(shell_script="/path/to/script.sh", script_args="arg1 arg2")
        for key, value in added_vars.items():
            print(f"{key} = {value}")
    """
    split_line = "HERE_IS_SPLIT"

    # Run the shell script and capture the environment variables
    variables = (
        subprocess.Popen(
            (
                "bash",
                "-c",
                f"set; source $1 $2 > /dev/null 2>&1; echo {split_line}; set",
                "_",
                shell_script,
                script_args,
            ),
            shell=False,
            stdout=subprocess.PIPE,
        )
        .communicate()[0]
        .decode("utf-8")
    )

    # Create dictionaries to store variables before and after script execution
    vars_before = dict()
    vars_after = dict()

    current_dict = vars_before
    for line in variables.split("\n"):
        if line == split_line:
            current_dict = vars_after
        else:
            spl = line.split("=", 1)

            if len(spl) > 1:
                current_dict[spl[0]] = spl[1].replace("'", "")
            elif len(spl[0].strip()) > 0:
                current_dict[spl[0]] = None

    # Identify added and changed variables by comparing before and after dictionaries
    added_or_changed_vars = {
        var: vars_after[var] for var in vars_after if var not in vars_before or vars_after[var] != vars_before[var]
    }

    return added_or_changed_vars


def changed_env_paths(shell_script):
    """Return env variables with the paths
    that have been changed or added
    """
    # It contains leftovers from functions
    added_vars = script_vars(shell_script)
    
    # Take only if contains paths, i.e. starts with "/"
    env_paths = dict()
    for key, val in added_vars.items():
        if isinstance(val, str) and val[0] == "/":
            env_paths[key] = val
    
    return env_paths        