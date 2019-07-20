USE [Home_Credit]
GO

DROP TABLE [dbo].[Pos_Cash_Balance_STG]
GO


CREATE TABLE [dbo].[Pos_Cash_Balance_STG](
	[SK_ID_PREV] [nvarchar](500) NULL,
	[SK_ID_CURR] [nvarchar](500) NULL,
	[MONTHS_BALANCE] [nvarchar](500) NULL,
	[CNT_INSTALMENT] [nvarchar](500) NULL,
	[CNT_INSTALMENT_FUTURE] [nvarchar](500) NULL,
	[NAME_CONTRACT_STATUS] [nvarchar](500) NULL,
	[SK_DPD] [nvarchar](500) NULL,
	[SK_DPD_DEF] [nvarchar](500) NULL
) ON [PRIMARY]
GO


