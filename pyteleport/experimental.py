from contextlib import contextmanager
import logging

from .teleport import tp_shell, pyteleport_skip_stack


@contextmanager
def allocate_disposable_ec2(service_name="ec2", region_name="us-east-1", ec2_resource_args=None,
                            create_instance_args=None, **template_kwargs):
    """
    Uses boto3 to allocate a disposable EC2 instance on AWS based on previously saved template.

    This context manager creates a new instance from a previously prepared
    launch template and ensures the instance is terminated with this context.

    Example usage: `with disposable_ec2(launch_template_id="lt-02d2bc621b78d5b8b") as instance: ...`.

    Tips:

    1. Prepare your launch template using AWS console or with a command line.
    2. Make sure the template is self-contained and specifies all required fields.
    3. Make sure to include your ssh key into the template.
    4. Make sure instance firewall rules include public IP address and allow ssh access.
    5. Make sure you install the same python version on your EC2 through user data.

    Parameters
    ----------
    service_name : str
        AWS service name.
    region_name : str
        AWS service region.
    ec2_resource_args
        Other arguments to `boto3.resource`.
    create_instance_args
        Other arguments to `EC2Resource.create_instances`.
    template_kwargs
        Arguments to EC2 template.

    Yields
    ------
        An `Instance` class representing the instance allocated.
    """
    import boto3

    if ec2_resource_args is None:
        ec2_resource_args = {}

    logging.info("Requesting EC2 instance ...")
    ec2 = boto3.resource(service_name=service_name, region_name=region_name, **ec2_resource_args)
    i_args = {
        "LaunchTemplate": template_kwargs,
        "MinCount": 1,
        "MaxCount": 1,
    }
    if create_instance_args is not None:
        i_args.update(create_instance_args)
    instance, = ec2.create_instances(**i_args)
    logging.info("Waiting to become online ...")
    instance.wait_until_running()
    logging.info("Reloading instance info ...")
    instance.load()
    try:
        logging.info(f"Instance {instance.id}")
        yield instance

    finally:
        logging.info(f"Terminating {instance.id}...")
        instance.terminate()


def tp_disposable_ec2(*args, allocate_kwargs=None, _skip=pyteleport_skip_stack(tp_shell), ec2_username="ec2-user",
                      ssh_retries=20, **kwargs):
    if allocate_kwargs is None:
        allocate_kwargs = {}
    with allocate_disposable_ec2(**allocate_kwargs) as ec2_instance:
        tp_shell("ssh",
                 "-o BatchMode=yes",  # do fail if password requested
                 "-o StrictHostKeyChecking=no",  # new key is expected
                 "-o UserKnownHostsFile=/dev/null",  # do not store to known hosts
                 f"-o ConnectionAttempts={ssh_retries}",  # ssh server may not be ready yet so continue attempting
                 "-R {port}:localhost:{port}",  # reverse tunnel for large object transmission
                 f"{ec2_username}@{ec2_instance.public_dns_name}",
                 "cloud-init status --wait > /dev/null;",  # wait for user data to complete (if any)
                 *args, _skip=_skip, **kwargs)
