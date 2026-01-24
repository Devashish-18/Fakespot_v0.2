export const generateExplanations = (data) => {
  const { profile_data, prediction } = data;
  const explanations = [];

  const isFake = prediction === 1 || prediction === 'FAKE';

  // Follower/Following ratio
  if (profile_data.following > 0) {
    const ratio = profile_data.following / profile_data.followers;
    if (ratio > 3) {
      explanations.push({
        metric: 'Follower/Following Ratio',
        explanation: `🚨 SUSPICIOUS: Following ${profile_data.following} accounts but only ${profile_data.followers} followers (ratio: ${ratio.toFixed(1)}:1). ${isFake ? 'This is a classic fake account pattern.' : 'While unusual, this could indicate an active account looking for engagement.'}`,
        severity: 'high'
      });
    } else if (ratio < 0.5) {
      explanations.push({
        metric: 'Follower/Following Ratio',
        explanation: `✓ POSITIVE: Healthy ratio with ${profile_data.followers} followers and only ${profile_data.following} following (ratio: ${ratio.toFixed(1)}:1). This is typical of established accounts with genuine audiences.`,
        severity: 'low'
      });
    }
  }

  // Account age
  if (profile_data.account_age_days < 30) {
    explanations.push({
      metric: 'Account Age',
      explanation: `⚠️ NEW ACCOUNT: Only ${profile_data.account_age_days} days old. ${isFake ? 'New accounts combined with other suspicious signals strongly indicate a fake profile.' : 'Being new is not necessarily problematic, but we monitor for other suspicious patterns.'}`,
      severity: 'high'
    });
  } else if (profile_data.account_age_days > 365) {
    explanations.push({
      metric: 'Account Age',
      explanation: `✓ ESTABLISHED: Account is ${profile_data.account_age_days} days old. Long-standing accounts are generally more trustworthy, indicating a genuine user.`,
      severity: 'low'
    });
  }

  // Engagement rate
  if (profile_data.engagement_rate < 0.01) {
    explanations.push({
      metric: 'Engagement Rate',
      explanation: `⚠️ LOW ENGAGEMENT: Only ${(profile_data.engagement_rate * 100).toFixed(2)}% engagement rate. ${isFake ? 'Extremely low engagement suggests bot followers with no real interaction.' : 'This could indicate inactive followers despite having a large follower count.'} With ${profile_data.followers} followers, we would expect higher interaction.`,
      severity: 'high'
    });
  } else if (profile_data.engagement_rate > 0.05) {
    explanations.push({
      metric: 'Engagement Rate',
      explanation: `✓ HEALTHY ENGAGEMENT: ${(profile_data.engagement_rate * 100).toFixed(2)}% engagement rate indicates an active, genuine audience. This is a strong indicator of account authenticity.`,
      severity: 'low'
    });
  }

  // Bio length
  if (profile_data.bio_length === 0) {
    explanations.push({
      metric: 'Biography',
      explanation: `⚠️ MISSING BIO: Account has no bio or description. ${isFake ? 'Lack of personal information is typical of fake accounts.' : 'Even authentic accounts usually have some description.'}`,
      severity: 'medium'
    });
  } else if (profile_data.bio_length > 50) {
    explanations.push({
      metric: 'Biography',
      explanation: `✓ DETAILED BIO: Account has a comprehensive bio (${profile_data.bio_length} characters), indicating a real user who wants to share information about themselves.`,
      severity: 'low'
    });
  }

  // Profile picture
  if (!profile_data.has_profile_pic) {
    explanations.push({
      metric: 'Profile Picture',
      explanation: `🚨 CRITICAL: No profile picture detected. ${isFake ? 'This is a major red flag - fake accounts rarely have legitimate profile pictures.' : 'Most legitimate accounts have profile pictures.'}`,
      severity: 'high'
    });
  } else {
    explanations.push({
      metric: 'Profile Picture',
      explanation: `✓ PROFILE PICTURE: Account has a profile picture, which is standard for legitimate accounts.`,
      severity: 'low'
    });
  }

  // Posts relative to followers
  if (profile_data.posts < 5 && profile_data.followers > 500) {
    explanations.push({
      metric: 'Content Activity',
      explanation: `🚨 MAJOR MISMATCH: Only ${profile_data.posts} posts but ${profile_data.followers} followers. ${isFake ? 'This severe mismatch is a hallmark of fake accounts that buy followers.' : 'This is unusual and may indicate artificial growth.'} A real account this large would have more content.`,
      severity: 'high'
    });
  } else if (profile_data.posts > 50) {
    explanations.push({
      metric: 'Content Activity',
      explanation: `✓ ACTIVE CREATOR: ${profile_data.posts} posts indicate a genuinely active user who regularly shares content. This is a strong authenticity indicator.`,
      severity: 'low'
    });
  }

  // Average likes
  if (profile_data.avg_likes < 2 && profile_data.posts > 0) {
    explanations.push({
      metric: 'Post Engagement',
      explanation: `⚠️ MINIMAL INTERACTION: Posts receive only ~${profile_data.avg_likes.toFixed(1)} likes on average. ${isFake ? 'Suggests the followers are bots with no genuine engagement.' : 'Even with a large following, posts should get more interaction.'}`,
      severity: 'medium'
    });
  } else if (profile_data.avg_likes > 10) {
    explanations.push({
      metric: 'Post Engagement',
      explanation: `✓ STRONG ENGAGEMENT: Posts receive ~${profile_data.avg_likes.toFixed(1)} likes on average, indicating real followers actively engaging with content.`,
      severity: 'low'
    });
  }

  // Private status
  if (profile_data.is_private) {
    explanations.push({
      metric: 'Account Status',
      explanation: `Account is set to private. While legitimate accounts can be private, fake accounts less commonly use this setting.`,
      severity: 'low'
    });
  } else {
    explanations.push({
      metric: 'Account Status',
      explanation: `Account is public, allowing open access to profile data. This is typical for both authentic and fake accounts.`,
      severity: 'low'
    });
  }

  return explanations;
};

export const generateChartExplanations = (data) => {
  const { profile_data, prediction, confidence } = data;
  const explanations = {};
  const isFake = prediction === 1 || prediction === 'FAKE';

  // Bar chart explanation - Metrics Overview
  explanations.bar = `📊 METRICS OVERVIEW: This account has ${profile_data.followers} followers, follows ${profile_data.following} accounts, and has ${profile_data.posts} posts. ${
    profile_data.following > profile_data.followers * 3
      ? `The extremely high following-to-follower ratio (${(profile_data.following / profile_data.followers).toFixed(1)}:1) is ${isFake ? 'a strong indicator of fake account behavior.' : 'a concerning signal that requires attention.'}`
      : profile_data.following > profile_data.followers
      ? `The account follows more than it's followed, which is unusual for established accounts.`
      : `The account maintains a healthy ratio with more followers than accounts followed, typical of authentic profiles.`
  }`;

  // Radar chart explanation - Health Indicators
  let radarIssues = [];
  let radarStrengths = [];

  if (profile_data.engagement_rate < 0.01) radarIssues.push('critically low engagement');
  else if (profile_data.engagement_rate > 0.05) radarStrengths.push('strong engagement');

  if (profile_data.account_age_days < 30) radarIssues.push('very new account');
  else if (profile_data.account_age_days > 365) radarStrengths.push('established history');

  if (profile_data.bio_length === 0) radarIssues.push('no biographical info');
  else if (profile_data.bio_length > 50) radarStrengths.push('detailed biography');

  if (!profile_data.has_profile_pic) radarIssues.push('missing profile picture');
  else radarStrengths.push('profile picture present');

  if (profile_data.posts < 5) radarIssues.push('minimal content');
  else if (profile_data.posts > 50) radarStrengths.push('active content creator');

  explanations.radar = `📈 ACCOUNT HEALTH: ${
    radarIssues.length > 0
      ? `Multiple concerns detected: ${radarIssues.join(', ')}. ${isFake ? 'Together, these signals strongly suggest an inauthentic account.' : 'These issues combined indicate questionable authenticity.'}`
      : radarStrengths.length > 0
      ? `Positive indicators detected: ${radarStrengths.join(', ')}. ${isFake ? 'However, other factors suggest account inauthenticity.' : 'These factors indicate a genuine, healthy account.'}`
      : 'Mixed signals across account health metrics.'
  }`;

  // Line chart explanation - Engagement Trend
  const totalEngagement = profile_data.avg_likes * profile_data.posts;
  explanations.line = `📉 ENGAGEMENT TREND: ${
    totalEngagement < 10
      ? `Minimal total engagement (${totalEngagement.toFixed(0)} estimated interactions across ${profile_data.posts} posts). ${isFake ? 'This indicates bot followers with zero real interaction.' : 'The account appears inactive or has artificially inflated follower count.'}`
      : totalEngagement < 100
      ? `Moderate engagement with ${profile_data.avg_likes.toFixed(1)} average likes per post. The engagement level is somewhat low relative to follower count.`
      : `Strong engagement pattern with ${profile_data.avg_likes.toFixed(1)} average likes per post across ${profile_data.posts} posts, indicating active genuine followers.`
  }`;

  // Doughnut explanation - Score Breakdown
  const profileScore = !profile_data.has_profile_pic || profile_data.bio_length === 0 ? 40 : 20;
  const networkScore = profile_data.following > profile_data.followers * 3 ? 50 : 25;
  const engagementScore = profile_data.engagement_rate < 0.01 ? 40 : 15;
  const authenticityScore = 100 - (profileScore + networkScore + engagementScore) / 3;

  explanations.doughnut = `🎯 PREDICTION BREAKDOWN (${Math.round(confidence * 100)}% confidence): ${
    isFake
      ? `This account is classified as FAKE due to: ${profileScore}% profile signals (missing info, picture, etc.), ${networkScore}% network signals (unusual follower ratios), and ${engagementScore}% engagement signals (low interaction). These factors combined indicate artificial account activity.`
      : `This account is classified as REAL due to: Healthy profile completeness (${100 - profileScore}%), normal network patterns (${100 - networkScore}%), and genuine engagement patterns (${100 - engagementScore}%). The overall authenticity score is ${Math.round(authenticityScore)}%, indicating a genuine user account.`
  }`;

  return explanations;
};

