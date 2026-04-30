# SAM3 Submodule

`deploy/third_party/sam3` 是一个真正的 git submodule，指向：

- `https://github.com/facebookresearch/sam3.git`

## 为什么用 submodule

- `sam3` 不是我们自己维护的代码。
- deploy 只依赖它的源码和 checkpoint 加载逻辑。
- 用 submodule 可以把上游源码版本固定到一个 commit，同时不把整份第三方历史直接并进 deploy 仓库。

## 首次 clone

如果你是第一次拿这份 `deploy` 仓库，推荐直接递归 clone：

```bash
git clone --recursive <deploy-repo-url>
```

如果你已经 clone 过了，但还没拉 submodule：

```bash
git submodule update --init --recursive
```

## 日常更新

查看 submodule 当前固定的 commit：

```bash
git submodule status
```

把 submodule 工作树同步到仓库里记录的 commit：

```bash
git submodule update --init --recursive
```

如果你明确想跟进上游最新提交：

```bash
git submodule update --remote --merge third_party/sam3
```

然后在 `deploy` 仓库根提交两部分变化：

- `.gitmodules`
- `third_party/sam3` 这个 gitlink 指针

## Python 安装

OpenPI venv 里需要把 submodule 以 editable 方式安装：

```bash
/home/edemlab/.local/bin/uv pip install \
  --python /home/edemlab/challenge_ws/baselines/openpi/.venv/bin/python \
  iopath \
  -e /home/edemlab/challenge_ws/deploy/third_party/sam3
```

如果之前 `sam3` 是从别的位置 editable 安装的，先卸载再重装更干净：

```bash
/home/edemlab/.local/bin/uv pip uninstall \
  --python /home/edemlab/challenge_ws/baselines/openpi/.venv/bin/python \
  sam3
```

然后再执行上面的 `uv pip install ... -e ...`

## Checkpoint

当前 deploy 代码默认读取：

- `/home/edemlab/challenge_ws/modelscope_cache/facebook/sam3___1/sam3.1_multiplex.pt`

submodule 只管理源码，不管理这个大权重文件。
