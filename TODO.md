# CS336 Assignment 1 전체 체크리스트

## A. 구현 (테스트로 검증됨 — `tests/adapters.py` 연결)

### 토크나이저
- [ ] `run_train_bpe` — byte-level BPE 학습 (15점) — special token 처리, 속도 제한 테스트 있음
- [ ] `get_tokenizer` — Tokenizer 클래스 (15점) — `encode`, `decode`, `encode_iterable`(메모리 1MB 제한)

### 모델 구성 요소
- [ ] `run_linear` — Linear 모듈 (1점)
- [ ] `run_embedding` — Embedding 모듈 (1점)
- [ ] `run_rmsnorm` — RMSNorm (1점)
- [ ] `run_silu` + `run_swiglu` — SwiGLU FFN (2점)
- [ ] `run_rope` — Rotary Positional Embedding (2점)
- [ ] `run_softmax` — 수치 안정 softmax (1점)
- [ ] `run_scaled_dot_product_attention` (5점)
- [ ] `run_multihead_self_attention` / `_with_rope` — causal MHA (5점)
- [ ] `run_transformer_block` — pre-norm Transformer 블록 (3점)
- [ ] `run_transformer_lm` — 전체 Transformer LM (3점)

### 학습 유틸리티
- [ ] `run_cross_entropy` — 수치 안정 cross-entropy (1점)
- [ ] `get_adamw_cls` — AdamW 옵티마이저 (2점)
- [ ] `run_get_lr_cosine_schedule` — warmup + cosine 스케줄
- [ ] `run_gradient_clipping` — 전체 L2 norm 클리핑 (1점)
- [ ] `run_get_batch` — 배치 샘플링 (2점)
- [ ] `run_save_checkpoint` / `run_load_checkpoint` (1점)

## B. 구현 (테스트 없음 — 과제 점수 항목)

- [ ] `decoding` — 생성 함수: temperature scaling + top-p(nucleus) sampling (3점)
- [ ] `training_together` — 학습 스크립트: 하이퍼파라미터 설정, `np.memmap` 데이터 로딩, 주기적 체크포인트·검증 손실 (4점)
- [ ] `experiment_log` — 실험 로깅 인프라 (step + wall-clock 기준 loss curve, wandb 등) (3점)
- [ ] 아블레이션 토글 — 모델에서 RMSNorm 제거 / post-norm / NoPE / SiLU FFN 전환 가능하게
- [ ] 데이터 준비 스크립트 — TinyStories/OWT 전체를 인코딩해 uint16 numpy 배열로 저장

## C. 학습/실험 (GPU 필요)

- [ ] 데이터 다운로드 — TinyStories + OWT 샘플 (`data/`)
- [ ] BPE 학습 실행 — TinyStories vocab 10K (2점), OWT vocab 32K (2점) + 분석 답변
- [ ] `learning_rate` — LR 튜닝, TinyStories 검증 손실 ≤ 1.45 달성 (2 B200시간, 3점) + 발산 LR 곡선 포함
- [ ] `batch_size_experiment` — 배치 크기 변화 실험 (1시간, 1점)
- [ ] `generate` — 학습된 모델로 256토큰 이상 생성 + 품질 논평 (1점)
- [ ] `layer_norm_ablation` — RMSNorm 제거 학습 곡선 (0.5시간, 1점)
- [ ] `pre_norm_ablation` — post-norm 학습 곡선 (0.5시간, 1점)
- [ ] `no_pos_emb` — NoPE vs RoPE 학습 곡선 (0.5시간, 1점)
- [ ] `swiglu_ablation` — SwiGLU vs SiLU 학습 곡선 (0.5시간, 1점)
- [ ] `main_experiment` — OWT 학습 + 곡선 + 생성 샘플 (2시간, 2점)
- [ ] `leaderboard` — OWT 검증 손실 최소화, 학습 45분 제한 (6점)

## D. 서면 답변 (writeup)

- [ ] `unicode1` — Unicode 이해 (1점)
- [ ] `unicode2` — UTF-8 인코딩 (3점)
- [ ] `tokenizer_experiments` — 압축률/처리량 분석 (4점)
- [ ] `transformer_accounting` — 파라미터 수/FLOPs 계산 (5점)
- [ ] `learning_rate_tuning` — SGD LR 실험 관찰 (1점)
- [ ] `adamw_accounting` — 메모리/FLOPs/학습 시간 계산 (2점)
- [ ] 실험 로그 문서 — 시도한 내용 전체 기록 (experiment_log 점수에 포함)
