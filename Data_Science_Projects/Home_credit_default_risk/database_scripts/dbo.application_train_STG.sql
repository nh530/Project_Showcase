﻿USE [Home_Credit]
GO


DROP TABLE IF EXISTS [dbo].[application_train_STG]
GO

CREATE TABLE [dbo].[application_train_STG](
	[SK_ID_CURR] [nvarchar](500) NULL,
	[TARGET] [nvarchar](500) NULL,
	[NAME_CONTRACT_TYPE] [nvarchar](500) NULL,
	[CODE_GENDER] [nvarchar](500) NULL,
	[FLAG_OWN_CAR] [nvarchar](500) NULL,
	[FLAG_OWN_REALTY] [nvarchar](500) NULL,
	[CNT_CHILDREN] [nvarchar](500) NULL,
	[AMT_INCOME_TOTAL] [nvarchar](500) NULL,
	[AMT_CREDIT] [nvarchar](500) NULL,
	[AMT_ANNUITY] [nvarchar](500) NULL,
	[AMT_GOODS_PRICE] [nvarchar](500) NULL,
	[NAME_TYPE_SUITE] [nvarchar](500) NULL,
	[NAME_INCOME_TYPE] [nvarchar](500) NULL,
	[NAME_EDUCATION_TYPE] [nvarchar](500) NULL,
	[NAME_FAMILY_STATUS] [nvarchar](500) NULL,
	[NAME_HOUSING_TYPE] [nvarchar](500) NULL,
	[REGION_POPULATION_RELATIVE] [nvarchar](500) NULL,
	[DAYS_BIRTH] [nvarchar](500) NULL,
	[DAYS_EMPLOYED] [nvarchar](500) NULL,
	[DAYS_REGISTRATION] [nvarchar](500) NULL,
	[DAYS_ID_PUBLISH] [nvarchar](500) NULL,
	[OWN_CAR_AGE] [nvarchar](500) NULL,
	[FLAG_MOBIL] [nvarchar](500) NULL,
	[FLAG_EMP_PHONE] [nvarchar](500) NULL,
	[FLAG_WORK_PHONE] [nvarchar](500) NULL,
	[FLAG_CONT_MOBILE] [nvarchar](500) NULL,
	[FLAG_PHONE] [nvarchar](500) NULL,
	[FLAG_EMAIL] [nvarchar](500) NULL,
	[OCCUPATION_TYPE] [nvarchar](500) NULL,
	[CNT_FAM_MEMBERS] [nvarchar](500) NULL,
	[REGION_RATING_CLIENT] [nvarchar](500) NULL,
	[REGION_RATING_CLIENT_W_CITY] [nvarchar](500) NULL,
	[WEEKDAY_APPR_PROCESS_START] [nvarchar](500) NULL,
	[HOUR_APPR_PROCESS_START] [nvarchar](500) NULL,
	[REG_REGION_NOT_LIVE_REGION] [nvarchar](500) NULL,
	[REG_REGION_NOT_WORK_REGION] [nvarchar](500) NULL,
	[LIVE_REGION_NOT_WORK_REGION] [nvarchar](500) NULL,
	[REG_CITY_NOT_LIVE_CITY] [nvarchar](500) NULL,
	[REG_CITY_NOT_WORK_CITY] [nvarchar](500) NULL,
	[LIVE_CITY_NOT_WORK_CITY] [nvarchar](500) NULL,
	[ORGANIZATION_TYPE] [nvarchar](500) NULL,
	[EXT_SOURCE_1] [nvarchar](500) NULL,
	[EXT_SOURCE_2] [nvarchar](500) NULL,
	[EXT_SOURCE_3] [nvarchar](500) NULL,
	[APARTMENTS_AVG] [nvarchar](500) NULL,
	[BASEMENTAREA_AVG] [nvarchar](500) NULL,
	[YEARS_BEGINEXPLUATATION_AVG] [nvarchar](500) NULL,
	[YEARS_BUILD_AVG] [nvarchar](500) NULL,
	[COMMONAREA_AVG] [nvarchar](500) NULL,
	[ELEVATORS_AVG] [nvarchar](500) NULL,
	[ENTRANCES_AVG] [nvarchar](500) NULL,
	[FLOORSMAX_AVG] [nvarchar](500) NULL,
	[FLOORSMIN_AVG] [nvarchar](500) NULL,
	[LANDAREA_AVG] [nvarchar](500) NULL,
	[LIVINGAPARTMENTS_AVG] [nvarchar](500) NULL,
	[LIVINGAREA_AVG] [nvarchar](500) NULL,
	[NONLIVINGAPARTMENTS_AVG] [nvarchar](500) NULL,
	[NONLIVINGAREA_AVG] [nvarchar](500) NULL,
	[APARTMENTS_MODE] [nvarchar](500) NULL,
	[BASEMENTAREA_MODE] [nvarchar](500) NULL,
	[YEARS_BEGINEXPLUATATION_MODE] [nvarchar](500) NULL,
	[YEARS_BUILD_MODE] [nvarchar](500) NULL,
	[COMMONAREA_MODE] [nvarchar](500) NULL,
	[ELEVATORS_MODE] [nvarchar](500) NULL,
	[ENTRANCES_MODE] [nvarchar](500) NULL,
	[FLOORSMAX_MODE] [nvarchar](500) NULL,
	[FLOORSMIN_MODE] [nvarchar](500) NULL,
	[LANDAREA_MODE] [nvarchar](500) NULL,
	[LIVINGAPARTMENTS_MODE] [nvarchar](500) NULL,
	[LIVINGAREA_MODE] [nvarchar](500) NULL,
	[NONLIVINGAPARTMENTS_MODE] [nvarchar](500) NULL,
	[NONLIVINGAREA_MODE] [nvarchar](500) NULL,
	[APARTMENTS_MEDI] [nvarchar](500) NULL,
	[BASEMENTAREA_MEDI] [nvarchar](500) NULL,
	[YEARS_BEGINEXPLUATATION_MEDI] [nvarchar](500) NULL,
	[YEARS_BUILD_MEDI] [nvarchar](500) NULL,
	[COMMONAREA_MEDI] [nvarchar](500) NULL,
	[ELEVATORS_MEDI] [nvarchar](500) NULL,
	[ENTRANCES_MEDI] [nvarchar](500) NULL,
	[FLOORSMAX_MEDI] [nvarchar](500) NULL,
	[FLOORSMIN_MEDI] [nvarchar](500) NULL,
	[LANDAREA_MEDI] [nvarchar](500) NULL,
	[LIVINGAPARTMENTS_MEDI] [nvarchar](500) NULL,
	[LIVINGAREA_MEDI] [nvarchar](500) NULL,
	[NONLIVINGAPARTMENTS_MEDI] [nvarchar](500) NULL,
	[NONLIVINGAREA_MEDI] [nvarchar](500) NULL,
	[FONDKAPREMONT_MODE] [nvarchar](500) NULL,
	[HOUSETYPE_MODE] [nvarchar](500) NULL,
	[TOTALAREA_MODE] [nvarchar](500) NULL,
	[WALLSMATERIAL_MODE] [nvarchar](500) NULL,
	[EMERGENCYSTATE_MODE] [nvarchar](500) NULL,
	[OBS_30_CNT_SOCIAL_CIRCLE] [nvarchar](500) NULL,
	[DEF_30_CNT_SOCIAL_CIRCLE] [nvarchar](500) NULL,
	[OBS_60_CNT_SOCIAL_CIRCLE] [nvarchar](500) NULL,
	[DEF_60_CNT_SOCIAL_CIRCLE] [nvarchar](500) NULL,
	[DAYS_LAST_PHONE_CHANGE] [nvarchar](500) NULL,
	[FLAG_DOCUMENT_2] [nvarchar](500) NULL,
	[FLAG_DOCUMENT_3] [nvarchar](500) NULL,
	[FLAG_DOCUMENT_4] [nvarchar](500) NULL,
	[FLAG_DOCUMENT_5] [nvarchar](500) NULL,
	[FLAG_DOCUMENT_6] [nvarchar](500) NULL,
	[FLAG_DOCUMENT_7] [nvarchar](500) NULL,
	[FLAG_DOCUMENT_8] [nvarchar](500) NULL,
	[FLAG_DOCUMENT_9] [nvarchar](500) NULL,
	[FLAG_DOCUMENT_10] [nvarchar](500) NULL,
	[FLAG_DOCUMENT_11] [nvarchar](500) NULL,
	[FLAG_DOCUMENT_12] [nvarchar](500) NULL,
	[FLAG_DOCUMENT_13] [nvarchar](500) NULL,
	[FLAG_DOCUMENT_14] [nvarchar](500) NULL,
	[FLAG_DOCUMENT_15] [nvarchar](500) NULL,
	[FLAG_DOCUMENT_16] [nvarchar](500) NULL,
	[FLAG_DOCUMENT_17] [nvarchar](500) NULL,
	[FLAG_DOCUMENT_18] [nvarchar](500) NULL,
	[FLAG_DOCUMENT_19] [nvarchar](500) NULL,
	[FLAG_DOCUMENT_20] [nvarchar](500) NULL,
	[FLAG_DOCUMENT_21] [nvarchar](500) NULL,
	[AMT_REQ_CREDIT_BUREAU_HOUR] [nvarchar](500) NULL,
	[AMT_REQ_CREDIT_BUREAU_DAY] [nvarchar](500) NULL,
	[AMT_REQ_CREDIT_BUREAU_WEEK] [nvarchar](500) NULL,
	[AMT_REQ_CREDIT_BUREAU_MON] [nvarchar](500) NULL,
	[AMT_REQ_CREDIT_BUREAU_QRT] [nvarchar](500) NULL,
	[AMT_REQ_CREDIT_BUREAU_YEAR] [nvarchar](500) NULL
) ON [PRIMARY]
GO


