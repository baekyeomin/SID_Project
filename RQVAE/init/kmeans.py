import numpy as np #kmeans 초기 centorid 무작위 선택시 넘파이 사용
import torch #tensor 연산 + 거리 계산 위해 파이토치 사용

from einops import rearrange #Tensor 차원을 재배치해서 pairwise distance를 계산하기 위해 사용
from typing import NamedTuple # K-means 결과를 centroids와 assignment로 묶어서 반환하기 위해 사용


def kmeans_init_(tensor: torch.Tensor, x: torch.Tensor):  # 입력 x에 K-means를 수행하고 결과 centroid로 tensor를 초기화하는 함수
    assert tensor.dim() == 2  # 초기화할 codebook tensor가 [코드 수, embedding 차원] 형태의 2차원인지 확인
    assert x.dim() == 2  # K-means 입력 x가 [데이터 수, embedding 차원] 형태의 2차원인지 확인

    with torch.no_grad(): # K-means 초기화 과정은 gradient 학습 대상이 아니므로 autograd 비활성화
        k, _ = tensor.shape  # codebook tensor의 첫 번째 차원에서 cluster 개수 k를 가져옴
        kmeans_out = Kmeans(k=k).run(x)  # x에 대해 k개의 cluster를 갖는 K-means 수행
        tensor.data.copy_(kmeans_out.centroids)  # 계산된 K-means centroid를 기존 codebook tensor 값으로 복사


class KmeansOutput(NamedTuple): # K-means 실행 결과를 저장
    centroids: torch.Tensor  # 최종적으로 계산된 각 cluster의 centroid
    assignment: torch.Tensor # 각 입력 데이터가 어떤 cluster에 배정되었는지를 나타내는 index


class Kmeans: # K-means clustering을 직접 수행하는 클래스
    def __init__(
        self, k: int, max_iters: int = None, stop_threshold: float = 1e-10
    ) -> None:
        self.k = k # 생성할 cluster의 개수 (k= 256)
        self.iters = max_iters  # K-means를 최대 몇 번 반복할지 지정, None이면 수렴할 때까지 반복
        self.stop_threshold = stop_threshold  # centroid 변화량이 이 값보다 작아지면 수렴했다고 판단
        self.centroids = None # 현재 centroid를 저장할 변수
        self.assignment = None # 각 데이터의 현재 cluster assignment를 저장할 변수

    def _init_centroids(self, x: torch.Tensor) -> None: # K-means 시작 시 초기 centroid를 정하는 함수
        B, D = x.shape # B는 입력 데이터 개수, D는 각 데이터 vector의 차원
        init_idx = np.random.choice(B, self.k, replace=False)  # B개 데이터 중 서로 다른 k개를 무작위로 선택
        self.centroids = x[init_idx, :] # 선택된 k개의 입력 vector를 초기 centroid로 설정
        self.assignment = None # 아직 cluster assignment는 계산하지 않았으므로 None으로 초기화

    def _update_centroids(self, x) -> torch.Tensor: # 현재 centroid를 기준으로 assignment와 centroid를 갱신하는 함수
        squared_pw_dist = (rearrange(x, "b d -> b 1 d") - rearrange(self.centroids, "b d -> 1 b d") ** 2  # 각 입력과 centroid의 차이를 제곱하여 차원별 squared distance 계산
        centroid_idx = (squared_pw_dist.sum(axis=2)).min(axis=1).indices # 각 입력에서 가장 가까운 centroid의 index 선택
        assigned = (rearrange(torch.arange(self.k, device=x.device), "d -> d 1") == centroid_idx) # 각 cluster별로 어떤 입력이 할당되었는지 Boolean matrix 생성

        for cluster in range(self.k): # 모든 cluster를 하나씩 순회하면서
            is_assigned_to_c = assigned[cluster] # 현재 cluster에 배정된 입력을 나타내는 Boolean
            if not is_assigned_to_c.any(): #현재 클러스터에 데이터가 0개 할당된경우 
                if x.size(0) > 0:
                    self.centroids[cluster, :] = x[torch.randint(0, x.size(0), (1,))].squeeze(0) # 무작위 입력 vector 하나를 뽑아 centroid로 사용
                else:
                    raise ValueError("Can not choose random element from x, x is empty") #입력 x 자체가 비어 있는 경우 centroid 선택 불가
            else:
                self.centroids[cluster, :] = x[is_assigned_to_c, :].mean(axis=0) #cluster에 1개 이상의 기사가 있는 경우 해당 cluster에 할당된 vector들의 평균을 새로운 centroid로 설정
        self.assignment = centroid_idx # 이번 iteration에서 계산한 cluster assignment 저장

#실제 k means 전체 과정 실행 함수 
    def run(self, x):
        self._init_centroids(x) # 입력 데이터에서 k개의 초기 centroid를 무작위 선택

        i = 0 # 현재 K-means iteration 횟수 초기화
        while self.iters is None or i < self.iters: # max_iters가 없거나 아직 최대 반복 횟수에 도달하지 않았다면 반복
            old_c = self.centroids.clone() # centroid 업데이트 전 값을 복사해서 저장
            self._update_centroids(x)# assignment를 다시 계산하고 centroid를 새로운 평균으로 갱신
            if torch.norm(self.centroids - old_c, dim=1).max() < self.stop_threshold: # 모든 centroid 중 가장 큰 이동량이 threshold보다 작은지 확인해서 
                break #거의 변하지 않으면 kmeans(centorid)가 수렴했다고 보고 종료 
            i += 1# K-means iteration 횟수 1 증가

        return KmeansOutput(centroids=self.centroids, assignment=self.assignment) # 최종 centroid와 각 데이터의 cluster assignment 리턴
