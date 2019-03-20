import os
import re
import subprocess
import base64
import os.path as osp
import pickle as pickle
import inspect
import sys
from rllab.misc import docker

import errno
import uuid

from rllab.core.serializable import Serializable
from rllab import config
from rllab.misc.console import mkdir_p
import datetime
import dateutil.tz
import json
import numpy as np

from rllab.misc.ext import AttrDict
from rllab.viskit.core import flatten
import collections


class StubBase(object):
    def __getitem__(self, item):
        return StubMethodCall(self, "__getitem__", args=[item], kwargs=dict())

    def __getattr__(self, item):
        try:
            return super(self.__class__, self).__getattribute__(item)
        except AttributeError:
            if item.startswith("__") and item.endswith("__"):
                raise
            return StubAttr(self, item)

    def __pow__(self, power, modulo=None):
        return StubMethodCall(self, "__pow__", [power, modulo], dict())

    def __call__(self, *args, **kwargs):
        return StubMethodCall(self.obj, self.attr_name, args, kwargs)

    def __add__(self, other):
        return StubMethodCall(self, "__add__", [other], dict())

    def __rmul__(self, other):
        return StubMethodCall(self, "__rmul__", [other], dict())

    def __div__(self, other):
        return StubMethodCall(self, "__div__", [other], dict())

    def __rdiv__(self, other):
        return StubMethodCall(BinaryOp(), "rdiv", [self, other], dict())  # self, "__rdiv__", [other], dict())

    def __rpow__(self, power, modulo=None):
        return StubMethodCall(self, "__rpow__", [power, modulo], dict())


class BinaryOp(Serializable):
    def __init__(self):
        Serializable.quick_init(self, locals())

    def rdiv(self, a, b):
        return b / a
        # def __init__(self, opname, a, b):
        #     self.opname = opname
        #     self.a = a
        #     self.b = b


class StubAttr(StubBase):
    def __init__(self, obj, attr_name):
        self.__dict__["_obj"] = obj
        self.__dict__["_attr_name"] = attr_name

    @property
    def obj(self):
        return self.__dict__["_obj"]

    @property
    def attr_name(self):
        return self.__dict__["_attr_name"]

    def __str__(self):
        return "StubAttr(%s, %s)" % (str(self.obj), str(self.attr_name))


class StubMethodCall(StubBase, Serializable):
    def __init__(self, obj, method_name, args, kwargs):
        self._serializable_initialized = False
        Serializable.quick_init(self, locals())
        self.obj = obj
        self.method_name = method_name
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        return "StubMethodCall(%s, %s, %s, %s)" % (
            str(self.obj), str(self.method_name), str(self.args), str(self.kwargs))


class StubClass(StubBase):
    def __init__(self, proxy_class):
        self.proxy_class = proxy_class

    def __call__(self, *args, **kwargs):
        if len(args) > 0:
            # Convert the positional arguments to keyword arguments
            spec = inspect.getargspec(self.proxy_class.__init__)
            kwargs = dict(list(zip(spec.args[1:], args)), **kwargs)
            args = tuple()
        return StubObject(self.proxy_class, *args, **kwargs)

    def __getstate__(self):
        return dict(proxy_class=self.proxy_class)

    def __setstate__(self, dict):
        self.proxy_class = dict["proxy_class"]

    def __getattr__(self, item):
        if hasattr(self.proxy_class, item):
            return StubAttr(self, item)
        raise AttributeError

    def __str__(self):
        return "StubClass(%s)" % self.proxy_class


class StubObject(StubBase):
    def __init__(self, __proxy_class, *args, **kwargs):
        if len(args) > 0:
            spec = inspect.getargspec(__proxy_class.__init__)
            kwargs = dict(list(zip(spec.args[1:], args)), **kwargs)
            args = tuple()
        self.proxy_class = __proxy_class
        self.args = args
        self.kwargs = kwargs

    def __getstate__(self):
        return dict(args=self.args, kwargs=self.kwargs, proxy_class=self.proxy_class)

    def __setstate__(self, dict):
        self.args = dict["args"]
        self.kwargs = dict["kwargs"]
        self.proxy_class = dict["proxy_class"]

    def __getattr__(self, item):
        # why doesnt the commented code work?
        # return StubAttr(self, item)
        # checks bypassed to allow for accesing instance fileds
        if hasattr(self.proxy_class, item):
            return StubAttr(self, item)
        raise AttributeError('Cannot get attribute %s from %s' % (item, self.proxy_class))

    def __str__(self):
        return "StubObject(%s, *%s, **%s)" % (str(self.proxy_class), str(self.args), str(self.kwargs))


class VariantDict(AttrDict):
    def __init__(self, d, hidden_keys):
        super(VariantDict, self).__init__(d)
        self._hidden_keys = hidden_keys

    def dump(self):
        return {k: v for k, v in self.items() if k not in self._hidden_keys}


class VariantGenerator(object):
    """
    Usage:

    vg = VariantGenerator()
    vg.add("param1", [1, 2, 3])
    vg.add("param2", ['x', 'y'])
    vg.variants() => # all combinations of [1,2,3] x ['x','y']

    Supports noncyclic dependency among parameters:
    vg = VariantGenerator()
    vg.add("param1", [1, 2, 3])
    vg.add("param2", lambda param1: [param1+1, param1+2])
    vg.variants() => # ..
    """

    def __init__(self):
        self._variants = []
        self._populate_variants()
        self._hidden_keys = []
        for k, vs, cfg in self._variants:
            if cfg.get("hide", False):
                self._hidden_keys.append(k)

    def add(self, key, vals, **kwargs):
        self._variants.append((key, vals, kwargs))

    def _populate_variants(self):
        methods = inspect.getmembers(
            self.__class__, predicate=lambda x: inspect.isfunction(x) or inspect.ismethod(x))
        methods = [x[1].__get__(self, self.__class__)
                   for x in methods if getattr(x[1], '__is_variant', False)]
        for m in methods:
            self.add(m.__name__, m, **getattr(m, "__variant_config", dict()))

    def variants(self, randomized=False):
        ret = list(self.ivariants())
        if randomized:
            np.random.shuffle(ret)
        return list(map(self.variant_dict, ret))

    def variant_dict(self, variant):
        return VariantDict(variant, self._hidden_keys)

    def to_name_suffix(self, variant):
        suffix = []
        for k, vs, cfg in self._variants:
            if not cfg.get("hide", False):
                suffix.append(k + "_" + str(variant[k]))
        return "_".join(suffix)

    def ivariants(self):
        dependencies = list()
        for key, vals, _ in self._variants:
            if hasattr(vals, "__call__"):
                args = inspect.getargspec(vals).args
                if hasattr(vals, 'im_self') or hasattr(vals, "__self__"):
                    # remove the first 'self' parameter
                    args = args[1:]
                dependencies.append((key, set(args)))
            else:
                dependencies.append((key, set()))
        sorted_keys = []
        # topo sort all nodes
        while len(sorted_keys) < len(self._variants):
            # get all nodes with zero in-degree
            free_nodes = [k for k, v in dependencies if len(v) == 0]
            if len(free_nodes) == 0:
                error_msg = "Invalid parameter dependency: \n"
                for k, v in dependencies:
                    if len(v) > 0:
                        error_msg += k + " depends on " + " & ".join(v) + "\n"
                raise ValueError(error_msg)
            dependencies = [(k, v)
                            for k, v in dependencies if k not in free_nodes]
            # remove the free nodes from the remaining dependencies
            for _, v in dependencies:
                v.difference_update(free_nodes)
            sorted_keys += free_nodes
        return self._ivariants_sorted(sorted_keys)

    def _ivariants_sorted(self, sorted_keys):
        if len(sorted_keys) == 0:
            yield dict()
        else:
            first_keys = sorted_keys[:-1]
            first_variants = self._ivariants_sorted(first_keys)
            last_key = sorted_keys[-1]
            last_vals = [v for k, v, _ in self._variants if k == last_key][0]
            if hasattr(last_vals, "__call__"):
                last_val_keys = inspect.getargspec(last_vals).args
                if hasattr(last_vals, 'im_self') or hasattr(last_vals, '__self__'):
                    last_val_keys = last_val_keys[1:]
            else:
                last_val_keys = None
            for variant in first_variants:
                if hasattr(last_vals, "__call__"):
                    last_variants = last_vals(
                        **{k: variant[k] for k in last_val_keys})
                    for last_choice in last_variants:
                        yield AttrDict(variant, **{last_key: last_choice})
                else:
                    for last_choice in last_vals:
                        yield AttrDict(variant, **{last_key: last_choice})


def variant(*args, **kwargs):
    def _variant(fn):
        fn.__is_variant = True
        fn.__variant_config = kwargs
        return fn

    if len(args) == 1 and isinstance(args[0], collections.Callable):
        return _variant(args[0])
    return _variant


def stub(glbs):
    # replace the __init__ method in all classes
    # hacky!!!
    for k, v in list(glbs.items()):
        # look at all variables that are instances of a class (not yet Stub)
        if isinstance(v, type) and v != StubClass:
            glbs[k] = StubClass(v)  # and replaces them by a the same but Stub


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


made_docker_image = False
docker_image = None


def make_docker_image(push=True):
    global made_docker_image
    global docker_image
    if not made_docker_image:
        docker_image = "{}:{}".format(config.DOCKER_IMAGE, str(uuid.uuid4()))
        docker.docker_build(docker_image)
        if push:
            docker.docker_push(docker_image)
        made_docker_image = True
    return docker_image


exp_count = 0
now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
remote_confirmed = False


def run_experiment_lite(
        stub_method_call=None,
        batch_tasks=None,
        exp_prefix="experiment",
        exp_name=None,
        log_dir=None,
        script="scripts/run_experiment_lite.py",
        mode="local",
        dry=False,
        docker_image=None,
        aws_config=None,
        env=None,
        variant=None,
        use_gpu=False,
        sync_s3_pkl=False,
        confirm_remote=True,
        terminate_machine=True,
        **kwargs):
    """
    Serialize the stubbed method call and run the experiment using the specified mode.
    :param stub_method_call: A stubbed method call.
    :param script: The name of the entrance point python script
    :param mode: Where & how to run the experiment. Should be one of "local", "local_docker", "ec2",
    and "lab_kube".
    :param dry: Whether to do a dry-run, which only prints the commands without executing them.
    :param exp_prefix: Name prefix for the experiments
    :param docker_image: name of the docker image. Ignored if using local mode.
    :param aws_config: configuration for AWS. Only used under EC2 mode
    :param env: extra environment variables
    :param kwargs: All other parameters will be passed directly to the entrance python script.
    :param variant: If provided, should be a dictionary of parameters
    """
    assert stub_method_call is not None or batch_tasks is not None, "Must provide at least either stub_method_call or batch_tasks"
    if batch_tasks is None:
        batch_tasks = [
            dict(
                kwargs,
                stub_method_call=stub_method_call,
                exp_name=exp_name,
                log_dir=log_dir,
                env=env,
                variant=variant
            )
        ]

    global exp_count
    global remote_confirmed
    config.USE_GPU = use_gpu

    # params_list = []

    for task in batch_tasks:
        call = task.pop("stub_method_call")
        data = base64.b64encode(pickle.dumps(call)).decode("utf-8")
        task["args_data"] = data
        exp_count += 1
        params = dict(kwargs)
        if task.get("exp_name", None) is None:
            task["exp_name"] = "%s_%s_%04d" % (
                exp_prefix, timestamp, exp_count)
        if task.get("log_dir", None) is None:
            task["log_dir"] = config.LOG_DIR + "/local/" + \
                              exp_prefix.replace("_", "-") + "/" + task["exp_name"]
        if task.get("variant", None) is not None:
            variant = task.pop("variant")
            if "exp_name" not in variant:
                variant["exp_name"] = task["exp_name"]
            task["variant_data"] = base64.b64encode(pickle.dumps(variant)).decode("utf-8")
        elif "variant" in task:
            del task["variant"]
        task["remote_log_dir"] = osp.join(
            config.AWS_S3_PATH, exp_prefix.replace("_", "-"), task["exp_name"])

    if mode not in ["local", "local_docker"] and not remote_confirmed and not dry and confirm_remote:
        remote_confirmed = query_yes_no(
            "Running in (non-dry) mode %s. Confirm?" % mode)
        if not remote_confirmed:
            sys.exit(1)

    if mode == "local":
        for task in batch_tasks:
            del task["remote_log_dir"]
            env = task.pop("env", None)
            command = to_local_command(
                task, script=osp.join(config.PROJECT_PATH, script), use_gpu=use_gpu)
            print(command)
            if dry:
                return
            try:
                if env is None:
                    env = dict()
                subprocess.call(
                    command, shell=True, env=dict(os.environ, **env))
            except Exception as e:
                print(e)
                if isinstance(e, KeyboardInterrupt):
                    raise
    elif mode == "local_docker":
        if docker_image is None:
            docker_image = config.DOCKER_IMAGE
        for task in batch_tasks:
            del task["remote_log_dir"]
            env = task.pop("env", None)
            command = to_docker_command(
                task,
                docker_image=docker_image,
                script=script,
                env=env,
                use_gpu=use_gpu,
            )
            print(command)
            if dry:
                return
            try:
                subprocess.call(command, shell=True)
            except Exception as e:
                if isinstance(e, KeyboardInterrupt):
                    raise

    elif mode == "lab_kube":

        assert env is None

        docker_image = make_docker_image(push=True)

        for task in batch_tasks:
            if 'env' in task:
                assert task.pop('env') is None
            task["resources"] = params.pop(
                "resouces", config.KUBE_DEFAULT_RESOURCES)
            task["node_selector"] = params.pop(
                "node_selector", config.KUBE_DEFAULT_NODE_SELECTOR)
            task["exp_prefix"] = exp_prefix
            pod_dict = to_lab_kube_pod(
                task, code_full_path="", docker_image=docker_image, script=script, is_gpu=use_gpu)
            pod_str = json.dumps(pod_dict, indent=1)
            if dry:
                print(pod_str)
            dir = "{pod_dir}/{exp_prefix}".format(
                pod_dir=config.POD_DIR, exp_prefix=exp_prefix)
            ensure_dir(dir)
            fname = "{dir}/{exp_name}.json".format(
                dir=dir,
                exp_name=task["exp_name"]
            )
            with open(fname, "w") as fh:
                fh.write(pod_str)
            kubecmd = "kubectl create -f %s" % fname
            print(kubecmd)
            if dry:
                return
            try:
                subprocess.call(kubecmd, shell=True)
            except Exception as e:
                if isinstance(e, KeyboardInterrupt):
                    raise
    else:
        raise NotImplementedError


_find_unsafe = re.compile(r'[a-zA-Z0-9_^@%+=:,./-]').search


def ensure_dir(dirname):
    """
    Ensure that a named directory exists; if it does not, attempt to create it.
    """
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def _shellquote(s):
    """Return a shell-escaped version of the string *s*."""
    if not s:
        return "''"

    if _find_unsafe(s) is None:
        return s

    # use single quotes, and put single quotes into double quotes
    # the string $'b is then quoted as '$'"'"'b'

    return "'" + s.replace("'", "'\"'\"'") + "'"


def _to_param_val(v):
    if v is None:
        return ""
    elif isinstance(v, list):
        return " ".join(map(_shellquote, list(map(str, v))))
    else:
        return _shellquote(str(v))


def to_local_command(params, script=osp.join(config.PROJECT_PATH, 'scripts/run_experiment.py'), use_gpu=False):
    command = "python " + script
    if use_gpu and not config.USE_TF:
        command = "THEANO_FLAGS='device=gpu,dnn.enabled=auto' " + command
    for k, v in params.items():
        if isinstance(v, dict):
            for nk, nv in v.items():
                if str(nk) == "_name":
                    command += "  --%s %s" % (k, _to_param_val(nv))
                else:
                    command += \
                        "  --%s_%s %s" % (k, nk, _to_param_val(nv))
        else:
            command += "  --%s %s" % (k, _to_param_val(v))
    return command


def to_docker_command(params, docker_image, script='scripts/run_experiment.py', pre_commands=None,
                      post_commands=None, dry=False, use_gpu=False, env=None, local_code_dir=None):
    """
    :param params: The parameters for the experiment. If logging directory parameters are provided, we will create
    docker volume mapping to make sure that the logging files are created at the correct locations
    :param docker_image: docker image to run the command on
    :param script: script command for running experiment
    :return:
    """
    log_dir = params.get("log_dir")
    # script = 'rllab/' + script
    if not dry:
        mkdir_p(log_dir)
    # create volume for logging directory
    if use_gpu:
        command_prefix = "nvidia-docker run"
    else:
        command_prefix = "docker run"
    docker_log_dir = config.DOCKER_LOG_DIR
    if env is not None:
        for k, v in env.items():
            command_prefix += " -e \"{k}={v}\"".format(k=k, v=v)
    command_prefix += " -v {local_mujoco_key_dir}:{docker_mujoco_key_dir}".format(
        local_mujoco_key_dir=config.MUJOCO_KEY_PATH, docker_mujoco_key_dir='/root/.mujoco')
    command_prefix += " -v {local_log_dir}:{docker_log_dir}".format(
        local_log_dir=log_dir,
        docker_log_dir=docker_log_dir
    )
    if local_code_dir is None:
        local_code_dir = config.PROJECT_PATH
    command_prefix += " -v {local_code_dir}:{docker_code_dir}".format(
        local_code_dir=local_code_dir,
        docker_code_dir=config.DOCKER_CODE_DIR
    )
    params = dict(params, log_dir=docker_log_dir)
    command_prefix += " -t " + docker_image + " /bin/bash -c "
    command_list = list()
    if pre_commands is not None:
        command_list.extend(pre_commands)
    command_list.append("echo \"Running in docker\"")
    command_list.append(to_local_command(
        params, osp.join(config.DOCKER_CODE_DIR, script), use_gpu=use_gpu))
    # We for 2 min sleep after termination to allow for last syncs.
    post_commands = ['sleep 120']
    if post_commands is not None:
        command_list.extend(post_commands)
    return command_prefix + "'" + "; ".join(command_list) + "'"


def dedent(s):
    lines = [l.strip() for l in s.split('\n')]
    return '\n'.join(lines)


S3_CODE_PATH = None


def s3_sync_code(config, dry=False):
    global S3_CODE_PATH
    if S3_CODE_PATH is not None:
        return S3_CODE_PATH
    base = config.AWS_CODE_SYNC_S3_PATH
    has_git = True
    try:
        current_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"]).strip()
        clean_state = len(
            subprocess.check_output(["git", "status", "--porcelain"])) == 0
    except subprocess.CalledProcessError as _:
        print("Warning: failed to execute git commands")
        has_git = False
    dir_hash = base64.b64encode(subprocess.check_output(["pwd"])).decode("utf-8")
    code_path = "%s_%s" % (
        dir_hash,
        (current_commit if clean_state else "%s_dirty_%s" % (current_commit, timestamp)) if
        has_git else timestamp
    )
    full_path = "%s/%s" % (base, code_path)
    cache_path = "%s/%s" % (base, dir_hash)
    cache_cmds = ["aws", "s3", "sync"] + \
                 [cache_path, full_path]
    cmds = ["aws", "s3", "sync"] + \
           flatten(["--exclude", "%s" % pattern] for pattern in config.CODE_SYNC_IGNORES) + \
           [".", full_path]
    caching_cmds = ["aws", "s3", "sync"] + \
                   [full_path, cache_path]
    mujoco_key_cmd = [
        "aws", "s3", "sync", config.MUJOCO_KEY_PATH, "{}/.mujoco/".format(base)]
    print(cache_cmds, cmds, caching_cmds, mujoco_key_cmd)
    if not dry:
        subprocess.check_call(cache_cmds)
        subprocess.check_call(cmds)
        subprocess.check_call(caching_cmds)
        try:
            subprocess.check_call(mujoco_key_cmd)
        except Exception:
            print('Unable to sync mujoco keys!')
    S3_CODE_PATH = full_path
    return full_path


def upload_file_to_s3(script_content):
    import tempfile
    import uuid
    f = tempfile.NamedTemporaryFile(delete=False)
    f.write(script_content)
    f.close()
    remote_path = os.path.join(
        config.AWS_CODE_SYNC_S3_PATH, "oversize_bash_scripts", str(uuid.uuid4()))
    subprocess.check_call(["aws", "s3", "cp", f.name, remote_path])
    os.unlink(f.name)
    return remote_path


def to_lab_kube_pod(
        params, docker_image, code_full_path,
        script='scripts/run_experiment.py', is_gpu=False
):
    """
    :param params: The parameters for the experiment. If logging directory parameters are provided, we will create
    docker volume mapping to make sure that the logging files are created at the correct locations
    :param docker_image: docker image to run the command on
    :param script: script command for running experiment
    :return:
    """
    log_dir = params.get("log_dir")
    remote_log_dir = params.pop("remote_log_dir")
    resources = params.pop("resources")
    node_selector = params.pop("node_selector")
    exp_prefix = params.pop("exp_prefix")
    mkdir_p(log_dir)
    pre_commands = list()
    pre_commands.append('mkdir -p ~/.aws')
    pre_commands.append('mkdir ~/.mujoco')
    # fetch credentials from the kubernetes secret file
    pre_commands.append('echo "[default]" >> ~/.aws/credentials')
    pre_commands.append(
        "echo \"aws_access_key_id = %s\" >> ~/.aws/credentials" % config.AWS_ACCESS_KEY)
    pre_commands.append(
        "echo \"aws_secret_access_key = %s\" >> ~/.aws/credentials" % config.AWS_ACCESS_SECRET)
    pre_commands.append('cd %s' % '/root/code/')
    pre_commands.append('mkdir -p %s' % log_dir)
    pre_commands.append("""
        while /bin/true; do
            aws s3 sync {log_dir} {remote_log_dir} --region {aws_region} --quiet
            sleep 15
        done & echo sync initiated""".format(log_dir=log_dir, remote_log_dir=remote_log_dir,
                                             aws_region=config.AWS_REGION_NAME))
    # copy the file to s3 after execution
    post_commands = list()
    post_commands.append('aws s3 cp --recursive %s %s' % (log_dir, remote_log_dir))
    command_list = list()
    if pre_commands is not None:
        command_list.extend(pre_commands)
    command_list.append("echo \"Running in docker\"")
    command_list.append(
        "%s 2>&1 | tee -a %s" % (
            to_local_command(params, script),
            "%s/stdouterr.log" % log_dir
        )
    )
    if post_commands is not None:
        command_list.extend(post_commands)
    command = "; ".join(command_list)
    pod_name = config.KUBE_PREFIX + params["exp_name"]
    # underscore is not allowed in pod names
    pod_name = pod_name.replace("_", "-")
    print("Is gpu: ", is_gpu)
    if not is_gpu:
        return {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": pod_name,
                "labels": {
                    "owner": config.LABEL,
                    "expt": pod_name,
                    "exp_time": timestamp,
                    "exp_prefix": exp_prefix,
                },
            },
            "spec": {
                "containers": [
                    {
                        "name": "foo",
                        "image": docker_image,
                        "command": [
                            "/bin/bash",
                            "-c",
                            "-li",  # to load conda env file
                            command,
                        ],
                        "resources": resources,
                        "imagePullPolicy": "Always",
                    }
                ],
                "restartPolicy": "Never",
                "nodeSelector": node_selector,
            }
        }
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": pod_name,
            "labels": {
                "owner": config.LABEL,
                "expt": pod_name,
                "exp_time": timestamp,
                "exp_prefix": exp_prefix,
            },
        },
        "spec": {
            "containers": [
                {
                    "name": "foo",
                    "image": docker_image,
                    "command": [
                        "/bin/bash",
                        "-c",
                        "-li",  # to load conda env file
                        command,
                    ],
                    "resources": resources,
                    "imagePullPolicy": "Always",
                    # gpu specific
                    "volumeMounts": [
                        {
                            "name": "nvidia",
                            "mountPath": "/usr/local/nvidia",
                            "readOnly": True,
                        }
                    ],
                    "securityContext": {
                        "privileged": True,
                    }
                }
            ],
            "volumes": [
                {
                    "name": "nvidia",
                    "hostPath": {
                        "path": "/var/lib/docker/volumes/nvidia_driver_352.63/_data",
                    }
                }
            ],
            "restartPolicy": "Never",
            "nodeSelector": node_selector,
        }
    }


def concretize(maybe_stub):
    if isinstance(maybe_stub, StubMethodCall):
        obj = concretize(maybe_stub.obj)
        method = getattr(obj, maybe_stub.method_name)
        args = concretize(maybe_stub.args)
        kwargs = concretize(maybe_stub.kwargs)
        return method(*args, **kwargs)
    elif isinstance(maybe_stub, StubClass):
        return maybe_stub.proxy_class
    elif isinstance(maybe_stub, StubAttr):
        obj = concretize(maybe_stub.obj)
        attr_name = maybe_stub.attr_name
        attr_val = getattr(obj, attr_name)
        return concretize(attr_val)
    elif isinstance(maybe_stub, StubObject):
        if not hasattr(maybe_stub, "__stub_cache"):
            args = concretize(maybe_stub.args)
            kwargs = concretize(maybe_stub.kwargs)
            try:
                maybe_stub.__stub_cache = maybe_stub.proxy_class(
                    *args, **kwargs)
            except Exception as e:
                print(("Error while instantiating %s" % maybe_stub.proxy_class))
                import traceback
                traceback.print_exc()
                # import ipdb; ipdb.set_trace()
        ret = maybe_stub.__stub_cache
        return ret
    elif isinstance(maybe_stub, dict):
        # make sure that there's no hidden caveat
        ret = dict()
        for k, v in maybe_stub.items():
            ret[concretize(k)] = concretize(v)
        return ret
    elif isinstance(maybe_stub, (list, tuple)):
        return maybe_stub.__class__(list(map(concretize, maybe_stub)))
    else:
        return maybe_stub
