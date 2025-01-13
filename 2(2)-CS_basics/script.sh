#!/bin/bash

# miniconda가 존재하지 않을 경우 설치
if ! command -v conda &> /dev/null; then
    echo "Miniconda is not installed. Starting installation..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    rm -f miniconda.sh
    echo "Miniconda installation completed."
else
    echo "Miniconda is already installed."
fi

# Conda 환경 생성 및 활성화
ENV_NAME="myenv"
if ! conda info --envs | grep "$ENV_NAME" &> /dev/null; then
    echo "Creating Conda environment '$ENV_NAME'..."
    conda create -y -n $ENV_NAME python=3.9
fi

echo "Activating Conda environment '$ENV_NAME'..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME


## 건드리지 마세요! ##
python_env=$(python -c "import sys; print(sys.prefix)")
if [[ "$python_env" == *"/envs/myenv"* ]]; then
    echo "가상환경 활성화: 성공"
else
    echo "가상환경 활성화: 실패"
    exit 1 
fi

# 필요한 패키지 설치
echo "Installing required packages..."
pip install mypy

# Submission 폴더 파일 실행
cd submission || { echo "submission 디렉토리로 이동 실패"; exit 1; }

for file in *.py; do
    # 문제 번호 추출
    problem_number=$(basename "$file" .py)

    # 단일 입력 파일 설정
    input_file="../input/${problem_number}_input"
    output_file="../output/${problem_number}_output"

    # 입력 파일이 존재하는지 확인
    if [[ -f "$input_file" ]]; then
        echo "Running $file..."
        python "$file" < "$input_file" > "$output_file"
        echo "Results saved in $output_file."
    else
        echo "Input file $input_file not found. Skipping $file."
    fi
done

# mypy 테스트 실행
echo "Running mypy test..."
mypy *.py

# 가상환경 비활성화
conda deactivate
echo "Task completed. Conda environment deactivated."