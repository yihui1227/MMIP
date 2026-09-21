import numpy as np
import pandas as pd


def generate_churn_dataset(n_samples=1000, random_state=42):
  np.random.seed(random_state)

  # 特徵生成
  tenure_months = np.random.randint(1, 48, size=n_samples)
  monthly_active_days = np.random.randint(0, 31, size=n_samples)
  support_tickets = np.random.poisson(lam=1.5, size=n_samples)
  avg_session_duration = np.round(
      np.random.gamma(shape=3.0, scale=8.0, size=n_samples), 1
  )
  overdue_payment_count = np.random.choice(
      [0, 1, 2, 3], size=n_samples, p=[0.75, 0.15, 0.07, 0.03]
  )
  plan_tier = np.random.choice([1, 2, 3], size=n_samples, p=[0.5, 0.35, 0.15])
  discount_applied = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])

  # 建立具備業務相關性的流失分數 (Log-odds)
  log_odds = (
      -0.5
      - 0.06 * tenure_months
      - 0.12 * monthly_active_days
      + 0.45 * support_tickets
      - 0.04 * avg_session_duration
      + 0.65 * overdue_payment_count
      - 0.30 * plan_tier
      + 0.25 * discount_applied
  )

  # 轉為機率並二值化
  prob = 1 / (1 + np.exp(-log_odds))
  is_churn = (np.random.rand(n_samples) < prob).astype(int)

  df = pd.DataFrame({
      'tenure_months': tenure_months,
      'monthly_active_days': monthly_active_days,
      'support_tickets_count': support_tickets,
      'avg_session_duration_min': avg_session_duration,
      'overdue_payment_count': overdue_payment_count,
      'plan_tier': plan_tier,
      'discount_applied': discount_applied,
      'is_churn': is_churn,
  })

  return df


# 產生 1,000 筆資料並輸出 CSV
df = generate_churn_dataset(1000)
df.to_csv('saas_churn_dataset.csv', index=False)
print(f'資料形狀: {df.shape}')
print(f"流失比例 (Class 1):\n{df['is_churn'].value_counts(normalize=True)}")