---
layout: post
title: "Privacy Policy — ArchitectureEncyclopedia Creator Upload Automation"
date: 2026-08-23
description: "Privacy Policy for the ArchitectureEncyclopedia Creator Upload Automation API client."
permalink: /blog/2026/Architecture-Encyclopedia-Privacy-Policy/
---

# Privacy Policy

**Last updated: August 23, 2026**

This Privacy Policy describes how **ArchitectureEncyclopedia Creator Upload Automation** (the **"API Client"**) accesses, uses, stores, and deletes information when using **YouTube API Services**.

ArchitectureEncyclopedia Creator Upload Automation is a privately operated Python application used by the owner of Architecture Encyclopedia to upload original videos to the owner's own YouTube channel. The API Client is not offered as a public service to third-party users.

By authorizing and using the API Client, the operator acknowledges this Privacy Policy.

- [Primary Access Page](https://utilmon.github.io/blog/2026/Architecture-Encyclopedia/)
- [Terms of Service](https://utilmon.github.io/blog/2026/Architecture-Encyclopedia-Terms-of-Service/)
- [Google Privacy Policy](https://policies.google.com/privacy)
- [YouTube Terms of Service](https://www.youtube.com/t/terms)

---

## 1. Use of YouTube API Services

The API Client uses the **YouTube Data API v3** and **Google OAuth 2.0** to obtain authorization to upload videos to the operator's YouTube channel.

The API Client is intended to request only the authorization required for its implemented functionality. For video uploading, this includes the YouTube OAuth scope:

`https://www.googleapis.com/auth/youtube.upload`

This permission allows the API Client to manage videos on the authorized YouTube account for the purpose described in this policy.

The API Client does **not** request or store the operator's Google or YouTube username and password.

---

## 2. Information Accessed and Processed

When the operator authorizes the API Client and uploads a video, the API Client may access or process the following information:

### Authorization information

The API Client may receive and use:

- OAuth 2.0 access tokens
- OAuth 2.0 refresh tokens, when offline/persistent authorization is enabled
- Authorization status associated with the operator's Google account

These credentials are used only to authenticate requests to YouTube API Services on behalf of the authorized operator.

### Video information provided by the operator

The API Client processes information supplied by the operator for upload, which may include:

- Video file
- Video title
- Video description
- Privacy status (`public`, `unlisted`, or `private`)
- Audience designation, including whether the video is made for kids when applicable
- Other upload metadata explicitly selected by the operator if supported by the client

### Information returned by YouTube API Services

The API Client may receive limited API data associated with an upload, such as:

- YouTube video ID
- Upload or processing status
- Video metadata returned as part of the upload response
- Error or API response information needed to complete or troubleshoot the upload

The API Client is not designed to collect YouTube viewing history, contacts, private messages, advertising profiles, or unrelated Google account information.

---

## 3. How Information Is Used

Information accessed through YouTube API Services is used only to provide the API Client's disclosed functionality, including:

1. Authenticating the operator through Google OAuth 2.0.
2. Uploading an operator-selected video to the authorized YouTube channel.
3. Applying the title, description, visibility, audience designation, and other metadata selected by the operator.
4. Reporting the result of the upload to the operator.
5. Maintaining authorization for later uploads when the operator has granted persistent authorization.
6. Diagnosing upload or authorization errors when necessary.

The API Client does not use YouTube API data for advertising, profiling, surveillance, credit decisions, data brokerage, or unrelated analytics.

---

## 4. User Control Over Upload Actions

The operator retains final control over actions performed by the API Client.

Before an upload is sent to YouTube, the API Client is intended to clearly identify the video and the metadata that will be submitted, including the selected privacy status.

The API Client does not intentionally change a video's privacy setting or other operator-provided upload values without the operator's instruction or consent.

---

## 5. Storage of Authorization Tokens and API Data

The API Client may store OAuth authorization tokens on the operator's local device so that repeated uploads can be authorized without requiring a new OAuth consent flow for every upload.

Authorization tokens are retained only for as long as needed for the authorized functionality and while the authorization remains active.

The API Client does not operate a public database for storing third-party users' YouTube data.

Other YouTube Authorized Data, if temporarily stored by the API Client, is retained only for as long as necessary for the authorized purpose and is deleted or refreshed in accordance with applicable YouTube API Services policies. Unless a longer period is specifically permitted by those policies, such stored Authorized Data is not retained for more than 30 calendar days without being deleted or refreshed.

Original video files and metadata created by the operator before an upload are the operator's own content and are not collected from YouTube API Services.

---

## 6. Sharing of Information

The API Client sends the operator's selected video, metadata, and authorization information to **Google/YouTube** only as necessary to perform authorized YouTube API operations.

The API Client does not sell, rent, or disclose Google user data or YouTube API data to data brokers, advertising platforms, or unrelated third parties.

No third-party users are given access to the operator's Authorized Data through this API Client.

Google and YouTube process information according to their own policies, including the [Google Privacy Policy](https://policies.google.com/privacy).

---

## 7. Cookies, Advertising, and Similar Technologies

The local Python API Client does not use cookies for advertising and does not permit third parties to serve advertisements through the API Client.

The API Client does not use YouTube API data for personalized advertising or retargeting.

The public documentation pages for this API Client are hosted separately as static web pages. The hosting provider may process ordinary technical information according to its own privacy practices. This Privacy Policy primarily describes the handling of Google user data and YouTube API data by the ArchitectureEncyclopedia Creator Upload Automation client.

---

## 8. Data Security

Reasonable technical measures are used to protect OAuth credentials and API data from unauthorized access, use, or disclosure.

OAuth credentials and client secrets are not intentionally published in public source code or exposed on the public documentation website.

Authorization tokens are used only for purposes consistent with the permission granted by the operator.

---

## 9. Revoking YouTube / Google Authorization

The operator can revoke the API Client's authorization at any time.

### Revoke through the API Client

The API Client provides a revoke/disconnect function that the operator can use to withdraw authorization. When invoked, the client requests revocation of the relevant OAuth token and stops using the authorization.

After revocation through the API Client, Authorized Data associated with that consent will be deleted as soon as possible and no later than **7 calendar days** after revocation.

### Revoke through Google Account settings

Authorization can also be revoked through Google's account security settings:

[Google Account — Third-party connections](https://security.google.com/settings/security/permissions)

After the operator revokes access through Google's security settings, API data associated with the revoked authorization will be deleted as soon as possible and no later than **30 calendar days** after the revocation is detected.

Revoking access prevents the API Client from making further authorized YouTube API requests unless authorization is granted again.

---

## 10. Requesting Deletion of Stored Data

Because the API Client is privately operated by the same person whose account is authorized, the operator controls the local environment where the API Client runs.

Stored OAuth tokens and locally stored YouTube API data can be deleted by:

1. Using the API Client's revoke/disconnect function.
2. Revoking the API Client through [Google Account security settings](https://security.google.com/settings/security/permissions).
3. Removing the API Client's locally stored credential/token data from the operator-controlled device.

Deletion requests and privacy questions may also be sent using the contact information in Section 13 below.

---

## 11. Data Retention After Authorization Ends

When authorization ends, the API Client will stop using the revoked authorization.

Authorized Data associated with a revocation initiated through the API Client will be deleted as soon as possible and within 7 calendar days.

When access is revoked through Google's security settings or the API Client can no longer verify valid authorization, related API data will be deleted as soon as possible and within the time limits required by YouTube API Services policies.

Authorization tokens that are no longer valid or necessary will not be intentionally retained indefinitely.

---

## 12. Changes to This Privacy Policy

This Privacy Policy may be updated if the API Client's functionality, data handling, or use of YouTube API Services changes.

If the API Client begins accessing, collecting, storing, sharing, or using Google user data in a materially different way, this Privacy Policy will be updated before that new use is implemented, and any additional consent required by Google or YouTube policies will be obtained.

The date at the top of this page indicates when this Privacy Policy was last updated.

---

## 13. Contact

Questions or complaints about this Privacy Policy or the API Client's privacy practices can be directed to the operator of ArchitectureEncyclopedia Creator Upload Automation.

**Privacy contact:** `ashedpipe@gmail.com`

**Website:** [ArchitectureEncyclopedia Creator Upload Automation](https://utilmon.github.io/blog/2026/Architecture-Encyclopedia/)

> **Before publishing:** Replace `ashedpipe@gmail.com` with an email address that you actively monitor. YouTube's Developer Policies require a way for users to contact the API Client owner or developer regarding privacy questions or complaints.

---

## 14. YouTube and Google Policies

Use of YouTube API Services is subject to applicable YouTube and Google policies.

For additional information, please review:

- [Google Privacy Policy](https://policies.google.com/privacy)
- [YouTube Terms of Service](https://www.youtube.com/t/terms)
- [YouTube API Services Terms of Service](https://developers.google.com/youtube/terms/api-services-terms-of-service)
- [YouTube API Services Developer Policies](https://developers.google.com/youtube/terms/developer-policies)

---

*ArchitectureEncyclopedia Creator Upload Automation is an independent application and is not endorsed by or affiliated with YouTube or Google.*
