# Git 日常维护速查

本项目远程仓库：

```text
git@github.com:yingjian-peng/rl_learn_notebook.git
```

## 一、第一次在另一台电脑使用

先确认电脑已经安装 Git，并且 GitHub SSH 公钥已经添加到自己的账号。

```bash
git clone git@github.com:yingjian-peng/rl_learn_notebook.git
cd rl_learn_notebook
```

检查是否连接成功：

```bash
git remote -v
git status
```

如果使用了带密码的 SSH 私钥，先加载密钥：

```bash
ssh-add ~/.ssh/id_ed25519_github_rl
ssh -T git@github.com
```

看到 `Hi yingjian-peng!` 一类的提示，说明 GitHub 认证成功。

## 二、每天开始工作前

先拉取另一台电脑可能已经推送的更新：

```bash
git pull --rebase
```

查看当前状态：

```bash
git status
```

## 三、每天结束工作时

查看修改了什么：

```bash
git diff
```

把修改加入暂存区：

```bash
git add .
```

也可以只添加指定文件：

```bash
git add 11_TRPO/trpo.py
```

再次确认将要提交的内容：

```bash
git status
git diff --cached
```

创建提交：

```bash
git commit -m "更新 TRPO 学习笔记"
```

推送到 GitHub：

```bash
git push
```

## 四、两台电脑之间的推荐流程

每次开始工作：

```bash
git pull --rebase
```

修改并测试代码后：

```bash
git add .
git commit -m "写一句本次修改的说明"
git push
```

提交信息可以参考：

```text
新增 REINFORCE 训练代码
完善 Actor-Critic 注释
修正 TRPO 参数更新
补充强化学习理论笔记
```

## 五、遇到冲突时

先查看冲突文件：

```bash
git status
```

打开冲突文件，处理下面这些标记：

```text
<<<<<<<
当前电脑的内容
=======
另一台电脑的内容
>>>>>>>
```

保留正确内容并删除冲突标记后，执行：

```bash
git add 冲突文件
git rebase --continue
```

处理完成后推送：

```bash
git push
```

如果想放弃本次合并过程：

```bash
git rebase --abort
```

## 六、几个常用查询命令

查看提交记录：

```bash
git log --oneline --decorate --graph -10
```

查看远程仓库：

```bash
git remote -v
```

查看当前分支：

```bash
git branch
```

查看某个文件的修改：

```bash
git diff -- 11_TRPO/trpo.py
```

查看最近一次提交：

```bash
git show --stat
```

## 七、撤销操作时要小心

取消某个文件的暂存，但保留文件修改：

```bash
git restore --staged 文件名
```

放弃某个文件尚未提交的修改：

```bash
git restore 文件名
```

第二条命令会丢弃该文件的本地修改，执行前先确认不再需要这些内容。

## 八、本工程当前电脑的特殊写法

当前工作区中的 `.git` 目录由运行环境占用，真正的 Git 元数据在 `.git-worktree`。
因此在当前这台电脑的项目根目录中，Git 命令需要加上：

```bash
GIT_DIR=.git-worktree git status
GIT_DIR=.git-worktree git pull --rebase
GIT_DIR=.git-worktree git add .
GIT_DIR=.git-worktree git commit -m "更新说明"
```

推送时使用本项目的 SSH 密钥：

```bash
GIT_SSH_COMMAND='ssh -i ~/.ssh/id_ed25519_github_rl -o IdentitiesOnly=yes' \
GIT_DIR=.git-worktree git push
```

在另一台通过 `git clone` 正常克隆出来的电脑上，不需要加 `GIT_DIR=.git-worktree`，
直接使用普通的 `git status`、`git pull`、`git commit` 和 `git push` 即可。

## 九、最常用的最短版本

开始工作：

```bash
git pull --rebase
```

保存并上传：

```bash
git add .
git commit -m "更新学习内容"
git push
```
