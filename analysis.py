import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# CSV를 EUC-KR 인코딩으로 읽기
df = pd.read_csv('data/seoul_daily_temperature_2023_2026.csv', encoding='euc-kr')

print("=" * 60)
print("📊 서울 일평균기온 데이터 분석 - 기초 데이터 검증 단계")
print("=" * 60)

# 1. 컬럼명
print("\n[1] 컬럼명")
print(f"    {list(df.columns)}")

# 2. 전체 행 수
print("\n[2] 전체 행 수")
print(f"    {len(df):,} 행")

# 3. 각 컬럼의 데이터 타입
print("\n[3] 각 컬럼의 데이터 타입")
for col in df.columns:
    print(f"    {col}: {df[col].dtype}")

# 4. 날짜 최소값 / 최대값 (아직 문자열 상태)
print("\n[4] 날짜 범위 (변환 전)")
print(f"    최소값: {df.iloc[:, 2].min()}")
print(f"    최대값: {df.iloc[:, 2].max()}")

# 5. 각 컬럼의 결측치 개수
print("\n[5] 각 컬럼의 결측치 개수")
for col in df.columns:
    missing_count = df[col].isnull().sum()
    print(f"    {col}: {missing_count}")

# 6. 날짜 중복 개수
# 일시 컬럼이 3번째 컬럼(인덱스 2)이라고 가정
date_col = df.iloc[:, 2]
duplicate_dates = date_col.duplicated().sum()
print("\n[6] 날짜 중복 개수")
print(f"    {duplicate_dates}개")

# 日時 컬럼을 날짜형으로 변환
df['일시'] = pd.to_datetime(df['일시'], format='%Y-%m-%d')

# 7. 분석 기간 내 빠진 날짜가 있는지
print("\n[7] 분석 기간 내 빠진 날짜 확인")
start_date = pd.to_datetime('2023-10-01')
end_date = pd.to_datetime('2026-09-30')
expected_days = (end_date - start_date).days + 1
actual_days = len(df)

print(f"    예상 일수: {expected_days}일")
print(f"    실제 일수: {actual_days}일")
print(f"    빠진 날짜: {expected_days - actual_days}일")

# 8. 평균기온의 최소값 / 최대값 / 평균값
temp_col = df.iloc[:, 3]  # 4번째 컬럼(인덱스 3): 평균기온
print("\n[8] 평균기온 통계")
print(f"    최소값: {temp_col.min()}°C")
print(f"    최대값: {temp_col.max()}°C")
print(f"    평균값: {temp_col.mean():.2f}°C")

# 변환된 데이터 미리보기
print("\n[9] 변환된 데이터 미리보기 (처음 5행)")
print(df.head())

print("\n" + "=" * 60)
print("✅ 데이터 검증 완료!")
print("=" * 60)

# =====================================================
# 이상치 검사 (IQR 방식)
# =====================================================

print("\n" + "=" * 60)
print("📊 이상치 검사 - IQR 방식")
print("=" * 60)

# Q1, Q3, IQR 계산
Q1 = df['평균기온(°C)'].quantile(0.25)
Q3 = df['평균기온(°C)'].quantile(0.75)
IQR = Q3 - Q1

# 하한, 상한 계산
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"\n[IQR 기반 이상치 판정 기준]")
print(f"    Q1 (25백분위수): {Q1:.2f}°C")
print(f"    Q3 (75백분위수): {Q3:.2f}°C")
print(f"    IQR: {IQR:.2f}°C")
print(f"    하한 (Q1 - 1.5×IQR): {lower_bound:.2f}°C")
print(f"    상한 (Q3 + 1.5×IQR): {upper_bound:.2f}°C")

# 이상치 판정
outliers = df[(df['평균기온(°C)'] < lower_bound) | (df['평균기온(°C)'] > upper_bound)]
outlier_count = len(outliers)

print(f"\n[이상치 검사 결과]")
print(f"    이상치 후보 개수: {outlier_count}개")

if outlier_count > 0:
    print(f"\n    이상치 후보 목록 (날짜, 평균기온):")
    for idx, row in outliers.iterrows():
        print(f"        {row['일시'].strftime('%Y-%m-%d')}: {row['평균기온(°C)']:.1f}°C")
    print(f"\n    ※ 주의: 이상치 후보는 삭제하지 않습니다.")
    print(f"       극값은 실제 기상 관측 결과이므로 유지하며,")
    print(f"       R REPORT 작성 시 별도 검토 대상입니다.")
else:
    print(f"    이상치 후보가 범위 내에 있습니다.")


# =====================================================
# 파생값 생성 단계
# =====================================================

print("\n" + "=" * 60)
print("📊 파생값 생성 - Q1, Q2, Q3의 분석 준비")
print("=" * 60)

# 데이터 정렬 (일시 기준)
df = df.sort_values('일시').reset_index(drop=True)

# 1. 연도, 월 추가 (Q1: 계절 패턴 분석)
df['연도'] = df['일시'].dt.year
df['월'] = df['일시'].dt.month

# 2. 이동평균 추가 (Q1: 계절 패턴 부드럽게 보기)
#    - 7일 이동평균: 주간 추세
#    - 30일 이동평균: 월간 추세
df['7일이동평균'] = df['평균기온(°C)'].rolling(window=7, center=True).mean()
df['30일이동평균'] = df['평균기온(°C)'].rolling(window=30, center=True).mean()

# 3. 전일 대비 기온 변화량 (Q2: 급격한 변화 발생일)
#    오늘 - 어제, 변화량이 크면 급격한 기온 변화
df['기온변화량'] = df['평균기온(°C)'].diff()

print("\n[새로 생성된 컬럼]")
print(f"    - 연도, 월: 계절 패턴 분석용")
print(f"    - 7일이동평균, 30일이동평균: 노이즈 제거 후 추세 확인")
print(f"    - 기온변화량: 전일 대비 기온 변화 (양수=상승, 음수=하강)")

# 변환된 데이터 미리보기
print("\n[파생값이 추가된 데이터 미리보기]")
print(df[['일시', '평균기온(°C)', '7일이동평균', '30일이동평균', '기온변화량']].head(10))

# =====================================================
# Q3: 월별 분석 (평균, 표준편차)
# =====================================================

print("\n" + "=" * 60)
print("📊 Q3: 월별 평균기온과 변동성 분석")
print("=" * 60)

# 연도-월 조합으로 그룹화
monthly_stats = df.groupby(['연도', '월'])['평균기온(°C)'].agg([
    ('월평균기온', 'mean'),
    ('표준편차', 'std'),
    ('최저기온', 'min'),
    ('최고기온', 'max'),
    ('일수', 'count')
]).reset_index()

# 연도-월을 보기 좋게 표현
monthly_stats['연월'] = monthly_stats['연도'].astype(str) + '-' + monthly_stats['월'].astype(str).str.zfill(2)

print("\n[월별 집계 결과 (처음 12개월)]")
print(monthly_stats[['연월', '월평균기온', '표준편차', '최저기온', '최고기온']].head(12).to_string(index=False))

# =====================================================
# 핵심 통계
# =====================================================

print("\n" + "=" * 60)
print("📊 핵심 확인 사항")
print("=" * 60)

print("\n[Q1: 계절 패턴]")
print(f"    월별 평균기온 범위: {monthly_stats['월평균기온'].min():.1f}°C ~ {monthly_stats['월평균기온'].max():.1f}°C")
print(f"    36개 연월 중 최고 평균기온: {monthly_stats.loc[monthly_stats['월평균기온'].idxmax(), '연월']} ({monthly_stats['월평균기온'].max():.1f}°C)")
print(f"    36개 연월 중 최저 평균기온: {monthly_stats.loc[monthly_stats['월평균기온'].idxmin(), '연월']} ({monthly_stats['월평균기온'].min():.1f}°C)")

print("\n[Q2: 급격한 기온 변화]")
max_temp_change = df['기온변화량'].abs().max()
max_change_date = df.loc[df['기온변화량'].abs().idxmax(), '일시']
max_change_value = df.loc[df['기온변화량'].abs().idxmax(), '기온변화량']
print(f"    가장 큰 기온 변화: {max_change_date.strftime('%Y-%m-%d')} ({max_change_value:+.1f}°C)")
print(f"    최대 변화폭: {max_temp_change:.1f}°C")

# 기온 변화량 절대값의 분포 분석
abs_change = df['기온변화량'].abs().dropna()
p90 = abs_change.quantile(0.90)
p95 = abs_change.quantile(0.95)
p99 = abs_change.quantile(0.99)

print(f"\n    [전일 대비 기온변화 절대값 분포]")
print(f"    90백분위수: {p90:.2f}°C")
print(f"    95백분위수: {p95:.2f}°C")
print(f"    99백분위수: {p99:.2f}°C")

# 5°C가 몇 백분위에 해당하는지 확인
large_change_count = (abs_change >= 5.0).sum()
ratio_large_change = (large_change_count / len(abs_change)) * 100
print(f"\n    [분석 기준: 5.0°C 이상의 급격한 변화]")
print(f"    이 기준은 데이터의 95백분위수(약 4.93°C)를 참고해 설정한 분석용 기준입니다.")
print(f"    발생 일수: {large_change_count}일 (전체 유효 변화량 {len(abs_change)}일 중 {ratio_large_change:.1f}%)")

print("\n[Q3: 월별 변동성]")
print(f"    월별 표준편차 범위: {monthly_stats['표준편차'].min():.2f}°C ~ {monthly_stats['표준편차'].max():.2f}°C")
max_std_idx = monthly_stats['표준편차'].idxmax()
min_std_idx = monthly_stats['표준편차'].idxmin()
max_std_row = monthly_stats.loc[max_std_idx]
min_std_row = monthly_stats.loc[min_std_idx]
print(f"    36개 연월 중 표준편차가 최대: {max_std_row['연월']} ({max_std_row['표준편차']:.2f}°C)")
print(f"    36개 연월 중 표준편차가 최소: {min_std_row['연월']} ({min_std_row['표준편차']:.2f}°C)")

print("\n" + "=" * 60)
print("✅ 파생값 생성 완료! Q1-Q3 분석 준비됨")
print("=" * 60)

# =====================================================
# 그래프 생성 단계
# =====================================================

print("\n" + "=" * 60)
print("📊 시각화 - Q1, Q2, Q3 그래프 생성")
print("=" * 60)

# figures 폴더 생성 확인
if not os.path.exists('figures'):
    os.makedirs('figures')

# =====================================================
# 그래프 1: Q1 - 계절 패턴 분석 (원본, 7일 이동평균, 30일 이동평균)
# =====================================================

fig, ax = plt.subplots(figsize=(14, 6))

# 데이터 플롯
ax.plot(df['일시'], df['평균기온(°C)'], label='Daily Mean Temp', alpha=0.5, linewidth=0.8, color='gray')
ax.plot(df['일시'], df['7일이동평균'], label='7-Day MA', linewidth=1.5, color='orange')
ax.plot(df['일시'], df['30일이동평균'], label='30-Day MA', linewidth=2, color='red')

# 레이아웃
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Temperature (°C)', fontsize=12)
ax.set_title('Q1: Seasonal Temperature Pattern (Oct 2023 - Sep 2026)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig('figures/01_temperature_trend.png', dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ figures/01_temperature_trend.png 저장됨")

# =====================================================
# 그래프 2: Q2 - 급격한 기온 변화 분석
# =====================================================

fig, ax = plt.subplots(figsize=(14, 6))

# 기온 변화량 플롯
colors = []
for val in df['기온변화량'].fillna(0):
    if abs(val) >= 5.0:
        colors.append('red')
    else:
        colors.append('steelblue')

ax.bar(df['일시'], df['기온변화량'], color=colors, width=1, alpha=0.7)

# 기준선
ax.axhline(y=0, color='black', linestyle='-', linewidth=1, label='Baseline')
ax.axhline(y=5, color='red', linestyle='--', linewidth=1.5, label='Threshold (±5°C)')
ax.axhline(y=-5, color='red', linestyle='--', linewidth=1.5)

# 범례 (수동으로 구성)
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='steelblue', alpha=0.7, label='|Change| < 5°C'),
    Patch(facecolor='red', alpha=0.7, label='|Change| >= 5°C'),
    plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label='Threshold (±5°C)')
]
ax.legend(handles=legend_elements, fontsize=11, loc='upper left')

# 레이아웃
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Temperature Change from Previous Day (°C)', fontsize=12)
ax.set_title('Q2: Daily Temperature Change Analysis', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig('figures/02_daily_temperature_change.png', dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ figures/02_daily_temperature_change.png 저장됨")

# =====================================================
# 그래프 3: Q3 - 월별 평균기온과 변동성
# =====================================================

# 월별 데이터: 1월~12월을 기준으로 3년 전체 집계
monthly_by_month = df.groupby('월')['평균기온(°C)'].agg(['mean', 'std', 'min', 'max']).reset_index()
monthly_by_month.columns = ['Month', 'Mean', 'Std', 'Min', 'Max']

fig, ax1 = plt.subplots(figsize=(12, 6))

# 기본: 월별 평균기온
x_pos = range(1, 13)
bars = ax1.bar(x_pos, monthly_by_month['Mean'], width=0.6, alpha=0.7, color='steelblue', label='Mean Temp')
ax1.set_xlabel('Month', fontsize=12)
ax1.set_ylabel('Mean Temperature (°C)', fontsize=12, color='steelblue')
ax1.set_title('Q3: Monthly Mean Temperature and Variability', fontsize=14, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
ax1.tick_params(axis='y', labelcolor='steelblue')
ax1.grid(True, alpha=0.3, axis='y')

# 오차 막대로 표준편차 표시
ax1.errorbar(x_pos, monthly_by_month['Mean'], yerr=monthly_by_month['Std'], 
             fmt='none', ecolor='red', elinewidth=2, capsize=5, capthick=1.5, alpha=0.7, label='Std Dev')

# 범례
ax1.legend(['Mean Temperature', 'Standard Deviation (±1σ)'], fontsize=11, loc='upper left')

plt.tight_layout()
plt.savefig('figures/03_monthly_mean_variability.png', dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ figures/03_monthly_mean_variability.png 저장됨")

print("\n" + "=" * 60)
print("✅ 모든 그래프 생성 완료!")
print("   - 01_temperature_trend.png (Q1: 계절 패턴)")
print("   - 02_daily_temperature_change.png (Q2: 급격한 변화)")
print("   - 03_monthly_mean_variability.png (Q3: 월별 변동성)")
print("=" * 60)
