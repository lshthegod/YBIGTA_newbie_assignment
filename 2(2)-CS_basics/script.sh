#!/bin/bash

# miniconda가 존재하지 않을 경우 설치
if ! command -v conda &> /dev/null; then
    echo "Miniconda가 설치되어 있지 않습니다. 설치를 시작합니다..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    rm -f miniconda.sh
    echo "Miniconda 설치 완료."
else
    echo "Miniconda가 이미 설치되어 있습니다."
fi

# Conda 환경 생성 및 활성화
ENV_NAME="myenv"
if ! conda info --envs | grep "$ENV_NAME" &> /dev/null; then
    echo "Conda 환경 '$ENV_NAME'을 생성 중입니다..."
    conda create -y -n $ENV_NAME python=3.9
fi

echo "Conda 환경 '$ENV_NAME'을 활성화합니다..."
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
echo "필요한 패키지를 설치합니다..."
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
        echo "$file을 실행 중입니다..."
        python "$file" < "$input_file" > "$output_file"
        echo "결과가 $output_file에 저장되었습니다."
    else
        echo "입력 파일 $input_file을 찾을 수 없습니다. $file 실행을 건너뜁니다."
    fi
done


# mypy 테스트 실행
echo "mypy 테스트를 실행합니다..."
mypy *.py

# 가상환경 비활성화
conda deactivate
echo "작업 완료. 가상환경 비활성화되었습니다."