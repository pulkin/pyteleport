pyteleport with AWS services
============================

Disposable EC2 instances
------------------------

The idea is to use pyteleport with disposable EC2 AWS instances with existing
in a short life cycle within pyteleport call.
`pyteleport.experimental` contains `tp_disposable_ec2` which, in turn, relies
on [boto3](https://github.com/boto/boto3) python interface to AWS API.

Pre-requisites
--------------

You should have an active [AWS account](https://aws.amazon.com/account/)
and have set up your access key to be able to use boto3 without credentials.
For example, this is how my `.aws/config` looks like:
```
[default]
aws_access_key_id=ACCESSKEY
aws_secret_access_key=somethinghereaswell
```

Next, you should set up an **EC2 launch template** for use with your pyteleport
setup.
Alternatively, you may specify all EC2 launch settings in the code (which is not
recommended).
I tested this on a free-tier **t2.micro** instance with **Ubuntu 22.04** where
python 3.10 comes by default (unlike Amazon linux coming with python 3.7).
Make sure you generated an **ssh key** and include it as a part of the template to
be able to access the generated instances as soon as they spawn.
Do not forget to allow incoming connections on port 22 in **firewall** settings.
Otherwise, you will also need to replicate your python environment remotely:
this can be done by specifying user data.
Mine looks like this:

```shell
#!/bin/bash
sudo apt update -y
sudo apt upgrade -y
sudo apt install python3-pip -y
pip install https://github.com/pulkin/pyteleport/archive/dev.tar.gz
```

Teleporting
-----------

Once the template is ready you may test it with `tp_disposable_ec2`.
Example call would look something like this:

```python
from pyteleport.experimental import tp_disposable_ec2

tp_disposable_ec2(
    allocate_kwargs={"LaunchTemplateId": "lt-02d2bc621b78d5b8b"},
    ec2_username="ubuntu", python="python3",
)
```

where `lt-02d2bc621b78d5b8b` is the template id you will figure out after
creating the template.

Debugging
---------

Setup verbose logging output by, for example, calling
`pyteleport.tests.helpers.setup_verbose_logging()`.

Among other possible issues, the module `dill` used by pyteleport for object
serialization cannot serialize `SSLContext` and `SSLSocket` types from
the built-in `ssl` library.
Since both are used by the `boto3` module they will cause runtime errors.
A quick fix is to replace these objects with `None`s as follows:

```python
import dill
from ssl import SSLContext, SSLSocket
for t in SSLContext, SSLSocket:
    dill.register(t)(lambda pickler, obj: pickler.save_reduce(lambda: None, tuple(), obj=obj))
```

Of course, this will break `boto3` objects that are in use (but not the module
itself).
