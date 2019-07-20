USE [Home_Credit]
GO


DROP TABLE IF EXISTS [dbo].[Credit_Card_Balance_STG]
GO



CREATE TABLE [dbo].[Credit_Card_Balance_STG](
	[SK_ID_PREV] [nvarchar](500) NULL,
	[SK_ID_CURR] [nvarchar](500) NULL,
	[MONTHS_BALANCE] [nvarchar](500) NULL,
	[AMT_BALANCE] [nvarchar](500) NULL,
	[AMT_CREDIT_LIMIT_ACTUAL] [nvarchar](500) NULL,
	[AMT_DRAWINGS_ATM_CURRENT] [nvarchar](500) NULL,
	[AMT_DRAWINGS_CURRENT] [nvarchar](500) NULL,
	[AMT_DRAWINGS_OTHER_CURRENT] [nvarchar](500) NULL,
	[AMT_DRAWINGS_POS_CURRENT] [nvarchar](500) NULL,
	[AMT_INST_MIN_REGULARITY] [nvarchar](500) NULL,
	[AMT_PAYMENT_CURRENT] [nvarchar](500) NULL,
	[AMT_PAYMENT_TOTAL_CURRENT] [nvarchar](500) NULL,
	[AMT_RECEIVABLE_PRINCIPAL] [nvarchar](500) NULL,
	[AMT_RECIVABLE] [nvarchar](500) NULL,
	[AMT_TOTAL_RECEIVABLE] [nvarchar](500) NULL,
	[CNT_DRAWINGS_ATM_CURRENT] [nvarchar](500) NULL,
	[CNT_DRAWINGS_CURRENT] [nvarchar](500) NULL,
	[CNT_DRAWINGS_OTHER_CURRENT] [nvarchar](500) NULL,
	[CNT_DRAWINGS_POS_CURRENT] [nvarchar](500) NULL,
	[CNT_INSTALMENT_MATURE_CUM] [nvarchar](500) NULL,
	[NAME_CONTRACT_STATUS] [nvarchar](500) NULL,
	[SK_DPD] [nvarchar](500) NULL,
	[SK_DPD_DEF] [nvarchar](500) NULL
) ON [PRIMARY]
GO


