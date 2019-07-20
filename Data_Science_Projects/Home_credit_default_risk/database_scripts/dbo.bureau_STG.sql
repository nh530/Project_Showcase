USE [Home_Credit]
GO


DROP TABLE IF EXISTS [dbo].[Bureau_STG]
GO


CREATE TABLE [dbo].[Bureau_STG](
	[SK_ID_CURR] [nvarchar](500) NULL,
	[SK_ID_BUREAU] [nvarchar](500) NULL,
	[CREDIT_ACTIVE] [nvarchar](500) NULL,
	[CREDIT_CURRENCY] [nvarchar](500) NULL,
	[DAYS_CREDIT] [nvarchar](500) NULL,
	[CREDIT_DAY_OVERDUE] [nvarchar](500) NULL,
	[DAYS_CREDIT_ENDDATE] [nvarchar](500) NULL,
	[DAYS_ENDDATE_FACT] [nvarchar](500) NULL,
	[AMT_CREDIT_MAX_OVERDUE] [nvarchar](500) NULL,
	[CNT_CREDIT_PROLONG] [nvarchar](500) NULL,
	[AMT_CREDIT_SUM] [nvarchar](500) NULL,
	[AMT_CREDIT_SUM_DEBT] [nvarchar](500) NULL,
	[AMT_CREDIT_SUM_LIMIT] [nvarchar](500) NULL,
	[AMT_CREDIT_SUM_OVERDUE] [nvarchar](500) NULL,
	[CREDIT_TYPE] [nvarchar](500) NULL,
	[DAYS_CREDIT_UPDATE] [nvarchar](500) NULL,
	[AMT_ANNUITY] [nvarchar](500) NULL
) ON [PRIMARY]
GO


