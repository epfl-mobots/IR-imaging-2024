import numpy as np
import pandas as pd
import imutils as im

import proplot as plot
import matplotlib.pyplot as plt


def plot_df_data(ax, df, title):
    cluster = np.array(df['cluster_contour'].to_list())
    core = np.array(df['core_contour'].to_list())
    cen_cluster = df['cluster_center_px'].to_list()
    cen_core = df['core_center_px'].to_list()

    ax.imshow(bg_img, cmap='gray')

    for c in cluster:
        ax.plot(c[0], c[1], c='c', alpha=0.01)
    print('a')

    for c in core:
        ax.plot(c[0], c[1], c='#ff9e03', alpha=0.02)
    print('b')

    for c in cen_cluster:
        ax.plot([c[0]], [c[1]], marker='o', ms=2, c='#0f4980', alpha=0.1)
    print('c')

    for c in cen_core:
        ax.plot([c[0]], [c[1]], marker='o', ms=2, c='#ff5703', alpha=0.1)
    print('d')

    ax.set_title(title)

rpi = "rpi2"
bg_img = im.load_bg_img('./outputs/2_zigzag/background/%s/'%rpi, scale_factor=0.6)

abc_area = 415 * 210

##### meteo data
df_contours = pd.read_pickle("1b_cluster_contours_%s_df.pkl" % rpi)

# https://www.timeanddate.com/sun/austria/graz?month=12&year=2020
sunrise_h = 7-1
sunset_h = 16-1
df_day = df_contours[(df_contours.index.hour >= sunrise_h) & (df_contours.index.hour <= sunset_h)]
df_night = df_contours[(df_contours.index.hour > sunset_h) | (df_contours.index.hour < sunrise_h)]

# fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(6, 10))
#
# plot_df_data(ax0, df_contours, "all data")
# plot_df_data(ax1, df_day, "day time")
# plot_df_data(ax2, df_night, "night time")
#
# # plt.gca().invert_yaxis()
# # plt.scatter(x_arr, y_arr)
# # ax = plt.gca()
# # ax.set_ylim(ax.get_ylim()[::-1])

max_cluster_area = df_contours['cluster_area_mm'].max()
max_core_area = df_contours['cluster_area_mm'].max()

acl = []
acr = []

num_days = 7
for i in range(num_days):
    df = df_contours[df_contours.index.day == i + 1]

    # acl.append([f/max_cluster_area for f in df['cluster_area_mm'].to_list()])
    # acr.append([f/max_core_area for f in df['core_area_mm'].to_list()])
    acl.append([f / abc_area for f in df['cluster_area_mm'].to_list()])
    acr.append([f / abc_area for f in df['core_area_mm'].to_list()])

acl = np.array(acl)
acr = np.array(acr)

acl_min = np.min(acl, axis=0)
acl_max = np.max(acl, axis=0)

acr_min = np.min(acr, axis=0)
acr_max = np.max(acr, axis=0)

acl = np.mean(acl, axis=0)
acr = np.mean(acr, axis=0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4), subplot_kw=dict(projection='polar'))

for i in range(num_days):
    df = df_contours[df_contours.index.day == i+1]

    h = np.array(df.index.hour.to_list())
    m = np.array(df.index.minute.to_list())
    m = m / 60
    h = h + m
    h_rads = (2*np.pi*h)/24

    # area_cluster = np.array(df['cluster_area_mm'].to_list()) / max_cluster_area
    # area_core = np.array(df['core_area_mm'].to_list()) / max_core_area
    area_cluster = np.array(df['cluster_area_mm'].to_list()) / abc_area
    area_core = np.array(df['core_area_mm'].to_list()) / abc_area

    # c = ('Qual2', i)
    c = '#cccccc'
    # ax.plot(h_rads, area_core   , marker='o', ms=2, color=c, alpha=1, lw=0.5)
    # ax.plot(h_rads, area_cluster, marker='o', ms=2, color=c, alpha=1, lw=0.5)
    ax1.scatter(h_rads, area_core   , marker='o', s=3, color='#e8dfb0', alpha=1)
    ax1.scatter(h_rads, area_cluster, marker='o', s=3, color=c, alpha=1)

ax1.plot(h_rads, acl    , marker='o', ms=2, color='k', alpha=1., lw=0.7, label='cluster area')
ax1.plot(h_rads, acr, marker='o', ms=2, color='#ff9500', alpha=1., lw=0.7, label='core area')

# ax2.plot(h_rads, acr_min, color='#ff9500', alpha=1., lw=0.5)
# ax2.plot(h_rads, acr_max, color='#ff9500', alpha=1., lw=0.5)
# ax2.plot(h_rads, acl_min, color='k', alpha=1., lw=0.5)
# ax2.plot(h_rads, acl_max, color='k', alpha=1., lw=0.5)

## Plot daylight part
sun_y_pos = 0.7
a = (2*np.pi*sunrise_h)/24
b = (2*np.pi*sunset_h)/24
ax1.scatter(a, sun_y_pos, marker='o', s=30, c='#f5dd40')
ax1.scatter(b, sun_y_pos, marker='o', s=30, c='#f5dd40')
ax1.plot( np.arange(a, b, 0.01), np.arange(a, b, 0.01).shape[0]*[sun_y_pos], c='#f5dd40', lw=1.0, label='day time')

#### ==================
ax2.scatter(a, sun_y_pos, marker='o', s=30, c='#f5dd40')
ax2.scatter(b, sun_y_pos, marker='o', s=30, c='#f5dd40')
ax2.plot( np.arange(a, b, 0.01), np.arange(a, b, 0.01).shape[0]*[sun_y_pos], c='#f5dd40', lw=1.0)


ax2.plot(h_rads, acr,  color='#ff9500', alpha=1., lw=0.7, label='core area')
ax2.plot(h_rads, acl,  color='k', alpha=.75, lw=0.7, label='cluster area')
# ax2.plot(h_rads, acr, marker='o', ms=2, color='#ff9500', alpha=1., lw=0.7)
# ax2.plot(h_rads, acl, marker='o', ms=2, color='k', alpha=.75, lw=0.7)

# Copy first item at the end of the data to fill the hole between the last and first data points
h_rads = np.append(h_rads, h_rads[0])
acr_min = np.append(acr_min, acr_min[0])
acr_max = np.append(acr_max, acr_max[0])
acl_min = np.append(acl_min, acl_min[0])
acl_max = np.append(acl_max, acl_max[0])

ax2.fill_between(h_rads, acr_max, acr_min, color='#e8dfb0', alpha=0.5, interpolate=True)
ax2.fill_between(h_rads, acl_max, acl_min, color=c, alpha=.5, interpolate=True)

# ax2.plot(h_rads, acr, marker='o', ms=2, color='#ff9500', alpha=1., lw=0.7)
# ax2.fill_between(h_rads,acr_max,acr_min, color='#e8dfb0', alpha=1,interpolate=True)
# ax2.plot(h_rads, acl    , marker='o', ms=2, color='k', alpha=.75, lw=0.7)
# ax2.fill_between(h_rads,acl_max,acl_min, color=c, alpha=.75, interpolate=True)


ax1.legend(loc="lower left", bbox_to_anchor=(1., 0.9))

ax1.set_theta_direction(-1)
ax1.set_theta_offset(np.pi/2.0)
ax1.set_xticks(np.linspace(0., 2.*np.pi, 24, endpoint=False))
ax1.set_xticklabels(range(24))
ax1.set_ylim(0, 0.75)
ax1.spines['polar'].set_visible(False)


ax2.set_theta_direction(-1)
ax2.set_theta_offset(np.pi/2.0)
ax2.set_xticks(np.linspace(0., 2.*np.pi, 24, endpoint=False))
ax2.set_xticklabels(range(24))
ax2.set_ylim(0, 0.75)
ax2.spines['polar'].set_visible(False)

# ax.set_yticks((0.300,0.350,0.400,0.450,0.500,0.550,0.600))
# ax.set_ylim([0.3,0.6])
# ax.set_yticklabels(('300','350','400','450','500','550','600'))

plt.tight_layout()
plt.show()
# plt.savefig('1b_cluster_area_time_day.png')
