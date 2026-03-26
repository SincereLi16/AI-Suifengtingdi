# Git 上传与拉取说明

远程仓库：<https://github.com/SincereLi16/AI-Suifengtingdi.git>  
默认分支：`main`

虚拟环境目录（如 `.venv_battle_ocr/`）已在 `.gitignore` 中，**勿**将整个 venv 拷进仓库；换电脑后在本机重新创建虚拟环境并安装依赖（见 `依赖与环境说明.txt`）。

---

## 另一台电脑：第一次获取代码

在要存放项目的目录打开终端，执行：

```bash
git clone https://github.com/SincereLi16/AI-Suifengtingdi.git
cd AI-Suifengtingdi
```

若本地文件夹名与仓库不同（例如仍使用「随风听笛6」），可克隆到指定目录名：

```bash
git clone https://github.com/SincereLi16/AI-Suifengtingdi.git 随风听笛6
cd 随风听笛6
```

---

## 已有仓库的电脑：拉取最新代码

进入项目根目录后：

```bash
git pull
```

建议在开始改代码前先执行一次，减少与远程冲突。

---

## 上传本地修改到 GitHub

进入项目根目录：

```bash
git add .
git commit -m "简要说明本次修改"
git push
```

若 `commit` 提示未设置姓名/邮箱，先执行（换成你自己的信息）：

```bash
git config --global user.name "你的名字"
git config --global user.email "你的邮箱"
```

仅当前仓库生效时，去掉 `--global`，并在项目目录下执行。

---

## 首次在本机关联远程（若尚未 `remote`）

若项目是拷贝而来、尚未绑定远程，可在项目根目录执行：

```bash
git remote add origin https://github.com/SincereLi16/AI-Suifengtingdi.git
git branch -M main
git push -u origin main
```

若已存在 `origin` 但地址不对，可改为：

```bash
git remote set-url origin https://github.com/SincereLi16/AI-Suifengtingdi.git
```

---

## 推送失败常见情况

- **认证**：GitHub HTTPS 推送需使用 **Personal Access Token**（或 SSH 密钥），不能使用账号密码。
- **远程已有新提交**：先 `git pull`（必要时 `git pull --rebase origin main`），解决冲突后再 `git push`。

---

## 参考

- Git 官方文档：<https://git-scm.com/doc>
