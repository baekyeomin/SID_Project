from modules.normalize import L2NormalizationLayer  # 출력 벡터에 L2 norm을 적용하기 위한 layer import
from typing import List  # hidden layer 차원들을 리스트 형태로 타입 지정
from torch import nn  # PyTorch의 신경망 layer와 Module을 사용하기 위해 import
from torch import Tensor  # 입력/출력 Tensor의 타입 지정을 위해 import


class MLP(nn.Module):  # MLP 정의 (파라미터는 gin에서)
    def __init__(  
        self,
        input_dim: int,  # 입력 벡터의 차원(예: 768)
        hidden_dims: List[int],  # 중간 hidden layer들의 차원, 예: [512, 256]
        out_dim: int,  # 출력 벡터의 차원(예: Encoder 출력 h(a)가 128차원이면 128)
        dropout: float = 0.0,  # hidden layer에 적용할 dropout 비율
        normalize: bool = False,  # 최종 출력에 L2 normalization을 적용 여부
    ) -> None:
        super().__init__()  # nn.Module의 초기화 함수를 호출해서 PyTorch 모델로

        self.input_dim = input_dim  # 전달받은 입력 차원을 객체 내부에 저장
        self.hidden_dims = hidden_dims  # 전달받은 hidden layer 차원들을 저장
        self.out_dim = out_dim  # 전달받은 최종 출력 차원을 저장
        self.dropout = dropout  # 전달받은 dropout 비율을 저장

        dims = [self.input_dim] + self.hidden_dims + [self.out_dim]  # 전체 layer 차원을 하나의 리스트로 연결

        self.mlp = nn.Sequential()  # Linear, ReLU, Dropout 등을 순서대로 담을 Sequential 모델 생성
        for i, (in_d, out_d) in enumerate(zip(dims[:-1], dims[1:])):  # 연속된 차원 쌍을 하나씩 꺼내 각 Linear layer 생성
            self.mlp.append(nn.Linear(in_d, out_d, bias=False))  # in_d 차원에서 out_d 차원으로 변환하는 Linear layer 추가
            if i != len(dims) - 2:  # 현재 layer가 마지막 출력 layer가 아닌 경우에만 activation 적용
                self.mlp.append(nn.ReLU())  # Linear 출력에 ReLU 비선형 활성화 함수 추가
                if dropout != 0:  
                    self.mlp.append(nn.Dropout(dropout))  # 지정된 비율만큼 hidden unit을 랜덤하게 비활성화하는 Dropout 추가
        self.mlp.append(L2NormalizationLayer() if normalize else nn.Identity())  # normalize=True면 최종 출력 L2 정규화, 아니면 아무 변화 없이 그대로 통과

    def forward(self, x: Tensor) -> Tensor:  # 입력 x를 실제 MLP에 통과시키는 forward 함수
        assert x.shape[-1] == self.input_dim, (  # 입력 Tensor의 마지막 차원이 설정된 input_dim과 같은지 확인
            f"Invalid input dim: Expected {self.input_dim}, found {x.shape[-1]}"  # 차원이 다르면 기대값과 실제값을 포함한 오류 메시지 출력
        )
        return self.mlp(x)  # 입력 x를 Sequential에 정의된 모든 layer에 순서대로 통과시킨 결과 반환