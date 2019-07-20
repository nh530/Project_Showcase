USE [Home_Credit]
GO


DROP TABLE IF EXISTS [dbo].[Previous_Application_STG]
GO

CREATE TABLE [dbo].[Previous_Application_STG](
	[SK_ID_PREV] [nvarchar](500) NULL,
	[SK_ID_CURR] [nvarchar](500) NULL,
	[NAME_CONTRACT_TYPE] [nvarchar](500) NULL,
	[AMT_ANNUITY] [nvarchar](500) NULL,
	[AMT_APPLICATION] [nvarchar](500) NULL,
	[AMT_CREDIT] [nvarchar](500) NULL,
	[AMT_DOWN_PAYMENT] [nvarchar](500) NULL,
	[AMT_GOODS_PRICE] [nvarchar](500) NULL,
	[WEEKDAY_APPR_PROCESS_START] [nvarchar](500) NULL,
	[HOUR_APPR_PROCESS_START] [nvarchar](500) NULL,
	[FLAG_LAST_APPL_PER_CONTRACT] [nvarchar](500) NULL,
	[NFLAG_LAST_APPL_IN_DAY] [nvarchar](500) NULL,
	[RATE_DOWN_PAYMENT] [nvarchar](500) NULL,
	[RATE_INTEREST_PRIMARY] [nvarchar](500) NULL,
	[RATE_INTEREST_PRIVILEGED] [nvarchar](500) NULL,
	[NAME_CASH_LOAN_PURPOSE] [nvarchar](500) NULL,
	[NAME_CONTRACT_STATUS] [nvarchar](500) NULL,
	[DAYS_DECISION] [nvarchar](500) NULL,
	[NAME_PAYMENT_TYPE] [nvarchar](500) NULL,
	[CODE_REJECT_REASON] [nvarchar](500) NULL,
	[NAME_TYPE_SUITE] [nvarchar](500) NULL,
	[NAME_CLIENT_TYPE] [nvarchar](500) NULL,
	[NAME_GOODS_CATEGORY] [nvarchar](500) NULL,
	[NAME_PORTFOLIO] [nvarchar](500) NULL,
	[NAME_PRODUCT_TYPE] [nvarchar](500) NULL,
	[CHANNEL_TYPE] [nvarchar](500) NULL,
	[SELLERPLACE_AREA] [nvarchar](500) NULL,
	[NAME_SELLER_INDUSTRY] [nvarchar](500) NULL,
	[CNT_PAYMENT] [nvarchar](500) NULL,
	[NAME_YIELD_GROUP] [nvarchar](500) NULL,
	[PRODUCT_COMBINATION] [nvarchar](500) NULL,
	[DAYS_FIRST_DRAWING] [nvarchar](500) NULL,
	[DAYS_FIRST_DUE] [nvarchar](500) NULL,
	[DAYS_LAST_DUE_1ST_VERSION] [nvarchar](500) NULL,
	[DAYS_LAST_DUE] [nvarchar](500) NULL,
	[DAYS_TERMINATION] [nvarchar](500) NULL,
	[NFLAG_INSURED_ON_APPROVAL] [nvarchar](500) NULL
) ON [PRIMARY]
GO


