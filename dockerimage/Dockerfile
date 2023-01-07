FROM nvcr.io/nvidia/nvhpc:22.3-devel-cuda_multi-ubuntu20.04

RUN apt update -y && apt install -y apt-utils curl sudo zsh && echo  "nvhpc-2022 hold" | dpkg --set-selections && apt upgrade -y
RUN useradd -m -s /bin/bash dockeruser && echo "dockeruser:dockeruser" | chpasswd && adduser dockeruser sudo
USER dockeruser
WORKDIR /home/dockeruser
RUN sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
COPY .zshrc /home/dockeruser/.zshrc
ENTRYPOINT /usr/bin/zsh
RUN type -p curl >/dev/null || (echo "dockeruser" | sudo -S apt install curl -y)
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg > temp1 && echo "dockeruser" | sudo -S dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg if=temp1 && rm temp1 \
&& sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
&& sudo apt update \
&& sudo apt install gh -y
