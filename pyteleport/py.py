import sys


module = __import__(f"py{sys.version_info[0]}_{sys.version_info[1]}", globals(), locals(), [], 1)
globals().update({
    i: getattr(module, i)
    for i in dir(module)
    if not i.startswith("_")
})

